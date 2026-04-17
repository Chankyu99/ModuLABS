"""
LLM 기반 위치 검증기
────────────────────
- Kalman이 뽑은 상위 후보 도시를 Gemini로 후검증한다.
- 목적은 "탐지"가 아니라 geocoding / country-level / org-as-city 오탐 제거다.
- 불확실한 경우에는 후보를 유지하고, 명확히 잘못된 경우만 제거한다.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import trafilatura
from bs4 import BeautifulSoup
from google import genai
from newspaper import Article as NewspaperArticle
# 병렬 처리를 위한 라이브러리
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline.city_utils import normalize_city_key, normalize_city_name
from pipeline.config import (
    CITY_BLACKLIST,
    LLM_ALLOW_SINGLE_ARTICLE_EXACT,
    LLM_ALLOW_ROI_PRIOR_SINGLE_SUPPORT,
    LLM_ALLOW_STRATEGIC_SINGLE_SUPPORT,
    LLM_CONFIDENCE_THRESHOLD,
    LLM_MIN_EXACT_SUPPORT,
    LLM_MODELS,
    LLM_PREFETCH_URLS,
    ROI_CITIES,
    LLM_STRATEGIC_KEYWORDS,
    LLM_TOP_K_URLS,
    LLM_TOP_N,
)


REQUEST_TIMEOUT = 7
MAX_BODY_CHARS = 4000
MAX_ARTICLES_PER_CANDIDATE = max(1, LLM_TOP_K_URLS)
MAX_PREFETCH_URLS = max(MAX_ARTICLES_PER_CANDIDATE, LLM_PREFETCH_URLS)
MODEL_CANDIDATES = list(dict.fromkeys([*LLM_MODELS, "gemini-2.5-flash"]))
ROI_CITY_KEYS = {normalize_city_key(city) for city in ROI_CITIES}

INVALID_LOCATION_TYPES = {
    "province_only",
    "country_only",
    "region_only",
    "org_not_city",
    "weapon_not_city",
    "actor_home_city",
}
VALID_LOCATION_TYPES = {"exact_city", "nearby_city"}

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "is_relevant_event",
        "location_validation",
        "resolved_location",
        "confidence",
        "reason",
        "event_summary_ko",
        "imagery_need_ko",
        "city_explicitly_mentioned",
        "attack_explicitly_mentioned",
        "target_explicitly_mentioned",
        "strategic_target",
        "target_category",
        "evidence_span",
    ],
    "properties": {
        "is_relevant_event": {"type": "boolean"},
        "location_validation": {
            "type": "string",
            "enum": sorted(
                VALID_LOCATION_TYPES
                | INVALID_LOCATION_TYPES
                | {"unclear"}
            ),
        },
        "resolved_location": {"type": "string"},
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "reason": {"type": "string"},
        "event_summary_ko": {"type": "string"},
        "imagery_need_ko": {"type": "string"},
        "city_explicitly_mentioned": {"type": "boolean"},
        "attack_explicitly_mentioned": {"type": "boolean"},
        "target_explicitly_mentioned": {"type": "boolean"},
        "strategic_target": {"type": "boolean"},
        "target_category": {"type": "string"},
        "evidence_span": {"type": "string"},
    },
}

SYSTEM_PROMPT = """You validate whether a GDELT candidate city is the true operational event location for satellite tasking.

Decide whether the candidate city is:
- the exact attacked / incident city,
- a nearby city in the same metro or operational area,
- only a province / region / country level mention,
- an organization / weapon / actor name incorrectly treated as a city,
- or unclear.

Rules:
- Prefer the actual attacked or incident location over the actor's home country.
- If the article only mentions a country or wide region, do not invent a city.
- "nearby_city" is allowed only when the article clearly points to the same operational area or adjacent city.
- If evidence is weak, return "unclear" with lower confidence instead of guessing.
- city_explicitly_mentioned should be true only when the article directly names the candidate or resolved city.
- attack_explicitly_mentioned should be true only when the article explicitly describes a strike / attack / blast / bombardment / interception / impact in that location.
- target_explicitly_mentioned should be true only when the article explicitly mentions the attacked facility, base, airport, port, bridge, plant, or other target in that location.
- strategic_target should be true only when the attacked target is a strategic military / nuclear / energy / port / airport / industrial facility worth satellite imaging.
- target_category should be a short label like airport, missile_base, refinery, nuclear_site, port, bridge, industrial_plant, residential_area, media_building, other.
- evidence_span should be a short quoted-like phrase or paraphrased snippet from the article showing the strongest evidence.
- Write event_summary_ko in Korean as one short sentence describing what happened in this location.
- Write imagery_need_ko in Korean as one short sentence describing why satellite imagery would be useful here.
- If evidence is weak or the location is invalid, still write concise Korean strings that explain the uncertainty or why imagery is low value.
- Return JSON only.
"""


@dataclass
class Article:
    url: str
    title: str
    source: str
    published_utc: str | None
    body: str


def load_env_key(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    env_path = Path(".env")
    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw = line.split("=", 1)
        if key.strip() == name:
            return raw.strip().strip("\"'")
    return None


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_domain(url: str) -> str:
    match = re.match(r"https?://([^/]+)", url or "")
    return match.group(1) if match else "unknown"


def get_meta_content(soup: BeautifulSoup, key: str, attr: str = "property") -> str:
    tag = soup.find("meta", attrs={attr: key})
    if tag and tag.get("content"):
        return clean_text(tag["content"])
    return ""


def parse_json_ld_article_data(soup: BeautifulSoup) -> dict[str, str]:
    result = {"headline": "", "description": "", "article_body": "", "date_published": ""}

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue

        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            candidates = [item]
            if isinstance(item.get("@graph"), list):
                candidates.extend([x for x in item["@graph"] if isinstance(x, dict)])
            for candidate in candidates:
                ctype = candidate.get("@type")
                if isinstance(ctype, list):
                    ctype = " ".join(ctype)
                ctype = str(ctype or "").lower()
                if "article" not in ctype and "newsarticle" not in ctype:
                    continue
                result["headline"] = result["headline"] or clean_text(candidate.get("headline", ""))
                result["description"] = result["description"] or clean_text(candidate.get("description", ""))
                result["article_body"] = result["article_body"] or clean_text(candidate.get("articleBody", ""))
                result["date_published"] = result["date_published"] or clean_text(candidate.get("datePublished", ""))
    return result


def filter_paragraphs(paragraphs: list[str], page_host: str) -> list[str]:
    bad_prefixes = (
        "copyright ",
        "all rights reserved",
        "sign up",
        "subscribe",
        "read more",
        "click here",
        "advertisement",
    )
    filtered = []
    for paragraph in paragraphs:
        text = clean_text(paragraph)
        lower = text.lower()
        if len(text) < 50:
            continue
        if lower.startswith(bad_prefixes):
            continue
        if page_host and page_host in lower and len(text) < 120:
            continue
        filtered.append(text)
    return filtered


def fetch_article(url: str) -> Article | None:
    if not url or not str(url).startswith(("http://", "https://")):
        return None

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SIA-LLM-Validator/1.0)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException:
        return None

    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    json_ld = parse_json_ld_article_data(soup)

    title = (
        get_meta_content(soup, "og:title")
        or get_meta_content(soup, "twitter:title", attr="name")
        or json_ld["headline"]
        or clean_text(soup.title.string if soup.title and soup.title.string else "")
    )
    source = (
        get_meta_content(soup, "og:site_name")
        or get_meta_content(soup, "application-name", attr="name")
        or extract_domain(url)
    )
    published = (
        get_meta_content(soup, "article:published_time")
        or get_meta_content(soup, "og:updated_time")
        or get_meta_content(soup, "pubdate", attr="name")
        or json_ld["date_published"]
        or None
    )

    lead = (
        get_meta_content(soup, "og:description")
        or get_meta_content(soup, "twitter:description", attr="name")
        or get_meta_content(soup, "description", attr="name")
        or json_ld["description"]
    )

    body_candidates: list[str] = []
    if lead:
        body_candidates.append(lead)
    if json_ld["article_body"]:
        body_candidates.append(json_ld["article_body"])

    try:
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
            url=url,
        )
        if extracted:
            body_candidates.append(clean_text(extracted))
    except Exception:
        pass

    try:
        news_article = NewspaperArticle(url=url, language="en")
        news_article.set_html(html)
        news_article.parse()
        if news_article.title:
            title = title or clean_text(news_article.title)
        if news_article.publish_date and not published:
            published = clean_text(news_article.publish_date.isoformat())
        if news_article.text:
            body_candidates.append(clean_text(news_article.text))
    except Exception:
        pass

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    article_tag = soup.find("article")
    if article_tag:
        body_candidates.append(article_tag.get_text(" ", strip=True))

    selectors = [
        "main",
        '[role="main"]',
        ".article-body",
        ".story-body",
        ".post-content",
        ".entry-content",
        ".article-content",
        ".article__body",
        ".article-text",
        ".article__content",
        ".story-text",
        ".news-content",
        ".content__body",
        ".body-copy",
        "#article-body",
        "#storytext",
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            body_candidates.append(node.get_text(" ", strip=True))

    host = urlparse(url).netloc.lower()
    paragraphs = filter_paragraphs([p.get_text(" ", strip=True) for p in soup.find_all("p")], host)
    if paragraphs:
        body_candidates.append(" ".join(paragraphs[:20]))

    body = clean_text(max(body_candidates, key=len, default=""))[:MAX_BODY_CHARS]
    if not title and not body:
        return None

    return Article(url=url, title=title, source=source, published_utc=published, body=body)


def build_article_bundle(articles: list[Article]) -> str:
    chunks = []
    for idx, article in enumerate(articles, start=1):
        chunks.append(
            f"""[Article {idx}]
source: {article.source}
url: {article.url}
published_utc: {article.published_utc}
title: {article.title}
body: {article.body}"""
        )
    return "\n\n".join(chunks)


def _candidate_city_variants(city: str) -> set[str]:
    raw = str(city).strip()
    norm = normalize_city_name(raw)
    variants = {
        raw,
        norm,
        raw.replace("-", " "),
        norm.replace("-", " "),
        raw.lower(),
        norm.lower(),
    }
    return {clean_text(v).lower() for v in variants if clean_text(v)}


def _article_city_score(article: Article, candidate_city: str) -> int:
    text = " ".join([article.title or "", article.body or ""]).lower()
    score = 0
    for variant in _candidate_city_variants(candidate_city):
        if not variant:
            continue
        if variant in text:
            score += 3
        compact = re.sub(r"[^a-z0-9]+", "", variant)
        text_compact = re.sub(r"[^a-z0-9]+", "", text)
        if compact and compact in text_compact:
            score += 1
    return score


def _article_target_score(article: Article) -> int:
    text = " ".join([article.title or "", article.body or ""]).lower()
    score = 0
    for keyword in LLM_STRATEGIC_KEYWORDS:
        if keyword in text:
            score += 2
    return score


def call_gemini(client: genai.Client, candidate_meta: dict[str, Any], article: Article) -> dict[str, Any]:
    prompt = f"""Validate this single article for operational satellite tasking.

Candidate metadata:
- candidate_city: {candidate_meta.get('city')}
- date: {candidate_meta.get('date')}
- conflict_index: {candidate_meta.get('conflict_index')}
- innovation_z: {candidate_meta.get('innov_z')}
- events: {candidate_meta.get('events')}
- candidate_lat: {candidate_meta.get('lat')}
- candidate_lon: {candidate_meta.get('lon')}
- actor_countries: {candidate_meta.get('actor_countries')}
- event_codes: {candidate_meta.get('event_codes')}

Article:
{build_article_bundle([article])}
"""

    last_error: Exception | None = None
    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "response_mime_type": "application/json",
                    "response_json_schema": SCHEMA,
                    "temperature": 0.1,
                    "thinking_config": {"thinking_budget": 0},
                },
            )
            text = getattr(response, "text", None)
            if not text:
                raise RuntimeError("empty Gemini response")
            return json.loads(text)
        except Exception as exc:
            last_error = exc
            time.sleep(1)

    if last_error is None:
        raise RuntimeError("all Gemini models failed")
    raise last_error


def extract_relevant_context(text: str, target_city: str) -> str:
    """파이썬 단계에서 기사 본문을 자르되, 못 찾으면 원문의 앞부분이라도 반환합니다."""
    # "Kuwait City" -> "Kuwait"처럼 첫 단어만 추출하여 검색 조건 완화
    target_lower = target_city.split()[0].lower()
    
    if text.lower().count(target_lower) < 1:
        # 지명 검색에 실패해도 빈손으로 보내지 않고, 기사 초반 1500자를 던져줍니다.
        return text[:1500] 

    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if len(paragraphs) > 1:
        relevant_paras = [p for p in paragraphs if target_lower in p.lower()]
        return "\n\n".join(relevant_paras[:2]) if relevant_paras else text[:1500]
    else:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        found_indices = [i for i, s in enumerate(sentences) if target_lower in s.lower()]
        used_indices = set()
        relevant_chunks = []
        for idx in found_indices[:2]:
            start, end = max(0, idx - 2), min(len(sentences), idx + 3)
            current_chunk = [i for i in range(start, end) if i not in used_indices]
            if current_chunk:
                relevant_chunks.append(" ".join([sentences[i] for i in current_chunk]))
                used_indices.update(current_chunk)
        return "\n\n[...] ".join(relevant_chunks) if relevant_chunks else text[:1500]

def build_gemini_prompt(target_city: str, articles_data: list, target_date: str) -> str:
    """여러 기사의 핵심 문단을 하나의 프롬프트로 묶어 AI에게 1번만 질문합니다."""
    articles_context = ""
    for idx, data in enumerate(articles_data, 1):
        articles_context += f"\n[Article {idx}]\nContent: {data['text']}\n"

    # DATE_MISMATCH 관련 로직을 모두 삭제하고, 타격 여부(SUCCESS, AMBIGUOUS, DROPPED)에만 집중하도록 프롬프트 최적화
    return f"""
    [Task] Evaluate military/conflict status for [{target_city}].
    [Context] Read the provided text chunks. Ignore external metadata. Datelines (reporter locations) are not conflict zones.

    [Input]
    {articles_context}

    [Definitions for Status]
    SUCCESS: Direct military attack or physical conflict explicitly occurred IN or ON [{target_city}].
    AMBIGUOUS: [{target_city}] is under indirect military tension (e.g., air raid sirens, evacuations, staging area) without direct strikes.
    DROPPED: [{target_city}] is merely a dateline, not the target, or the text completely lacks evidence of an attack.

    [Instructions for Message]
    If Status is SUCCESS or AMBIGUOUS: Write a 1-line summary specifying WHO did WHAT to WHOM in Korean.
    If Status is DROPPED: Write a 1-line reason explaining why it was rejected in Korean.

    [JSON Schema]
    {{
        "status": "SUCCESS" | "AMBIGUOUS" | "DROPPED",
        "message": "Output the summary or rejection reason here based on the status"
    }}
    """

def _filter_blacklist(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "city" not in results.columns:
        return results

    blocked = {normalize_city_key(name) for name in CITY_BLACKLIST}
    city_norm = results["city"].fillna("").astype(str).map(normalize_city_key)
    keep_mask = ~city_norm.isin(blocked)
    return results.loc[keep_mask].copy()


def _prepare_raw_events(raw_df: pd.DataFrame, url_df: pd.DataFrame | None) -> pd.DataFrame:
    raw = raw_df.copy()
    raw["date"] = raw["SQLDATE"].astype(str).str[:8]
    raw["city_norm"] = raw["ActionGeo_FullName"].astype(str).map(normalize_city_key)
    if "SOURCEURL" not in raw.columns or raw["SOURCEURL"].isna().all():
        raw["SOURCEURL"] = None

    if url_df is not None and not url_df.empty:
        mapping = url_df.copy()
        mapping["GLOBALEVENTID"] = mapping["GLOBALEVENTID"].astype(str)
        raw["GLOBALEVENTID"] = raw["GLOBALEVENTID"].astype(str)
        raw = raw.merge(
            mapping[["GLOBALEVENTID", "SOURCEURL"]].rename(columns={"SOURCEURL": "MAPPED_SOURCEURL"}),
            on="GLOBALEVENTID",
            how="left",
        )
        raw["SOURCEURL"] = raw["SOURCEURL"].fillna(raw["MAPPED_SOURCEURL"])
        raw = raw.drop(columns=["MAPPED_SOURCEURL"])

    for col in ["NumSources", "NumMentions", "NumArticles", "EventCode"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    return raw


def _candidate_weight(raw_events: pd.DataFrame) -> pd.Series:
    return (
        raw_events["NumSources"].fillna(0) * 4
        + raw_events["NumMentions"].fillna(0)
        + raw_events["NumArticles"].fillna(0)
    )


def _select_candidate_urls(raw_events: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    if raw_events.empty:
        return [], {}

    ranked = raw_events.copy()
    ranked["candidate_weight"] = _candidate_weight(ranked)
    ranked = ranked.sort_values(
        ["candidate_weight", "NumSources", "NumMentions"],
        ascending=[False, False, False],
    )

    candidate_city = normalize_city_name(str(ranked["ActionGeo_FullName"].astype(str).iloc[0])) if "ActionGeo_FullName" in ranked.columns and not ranked.empty else ""
    urls = []
    for url in ranked["SOURCEURL"].fillna("").astype(str):
        if not url or url in urls:
            continue
        if not url.startswith(("http://", "https://")):
            continue
        urls.append(url)
        if len(urls) >= MAX_PREFETCH_URLS:
            break

    event_codes = (
        ranked["EventCode"]
        .dropna()
        .astype(int)
        .astype(str)
        .head(8)
        .tolist()
    )
    actor_countries = sorted(
        {
            str(value)
            for col in ["Actor1CountryCode", "Actor2CountryCode"]
            if col in ranked.columns
            for value in ranked[col].dropna().astype(str).tolist()
            if value and value != "nan"
        }
    )

    meta = {
        "event_codes": event_codes,
        "actor_countries": actor_countries,
    }
    # URL slug 자체에 도시명이 있으면 약간 우선순위를 준다.
    urls = sorted(
        urls,
        key=lambda url: (
            -sum(1 for variant in _candidate_city_variants(candidate_city) if variant and variant.replace(" ", "-") in url.lower()),
            urls.index(url),
        ),
    )

    return urls, meta


def _is_strong_exact(output: dict[str, Any]) -> bool:
    return (
        bool(output.get("is_relevant_event", True))
        and str(output.get("location_validation", "") or "") == "exact_city"
        and bool(output.get("city_explicitly_mentioned", False))
        and bool(output.get("attack_explicitly_mentioned", False))
    )


def _is_strong_nearby(output: dict[str, Any]) -> bool:
    return (
        bool(output.get("is_relevant_event", True))
        and str(output.get("location_validation", "") or "") == "nearby_city"
        and bool(output.get("city_explicitly_mentioned", False))
        and bool(output.get("attack_explicitly_mentioned", False))
    )


def _is_strong_invalid(output: dict[str, Any]) -> bool:
    return (
        str(output.get("location_validation", "") or "") in INVALID_LOCATION_TYPES
        and (
            bool(output.get("city_explicitly_mentioned", False))
            or bool(output.get("attack_explicitly_mentioned", False))
            or float(output.get("confidence", 0.0) or 0.0) >= LLM_CONFIDENCE_THRESHOLD
        )
    )


def _is_strategic_supported(output: dict[str, Any]) -> bool:
    validation = str(output.get("location_validation", "") or "")
    return (
        bool(output.get("is_relevant_event", True))
        and bool(output.get("strategic_target", False))
        and bool(output.get("attack_explicitly_mentioned", False))
        and bool(output.get("target_explicitly_mentioned", False))
        and validation in VALID_LOCATION_TYPES
    )


def _is_roi_prior_supported(output: dict[str, Any], row: pd.Series) -> bool:
    city_key = normalize_city_key(row.get("city", ""))
    validation = str(output.get("location_validation", "") or "")
    if city_key not in ROI_CITY_KEYS:
        return False
    return (
        bool(output.get("is_relevant_event", True))
        and validation in VALID_LOCATION_TYPES
        and bool(output.get("attack_explicitly_mentioned", False))
        and (
            bool(output.get("city_explicitly_mentioned", False))
            or bool(output.get("target_explicitly_mentioned", False))
        )
    )


def _aggregate_article_outputs(article_outputs: list[dict[str, Any]], row: pd.Series) -> dict[str, Any]:
    if not article_outputs:
        return {
            "llm_keep": True,
            "llm_actionability": "unverified",
            "llm_validation_type": "unverified",
            "llm_confidence": -1.0,
            "llm_reason": "LLM 검증용 기사 본문을 확보하지 못해 원본 후보를 유지했습니다.",
            "llm_resolved_city": str(row["city"]),
            "llm_event_summary": "",
            "llm_imagery_need": "",
            "llm_article_count": 0,
            "llm_exact_support": 0,
            "llm_nearby_support": 0,
            "llm_invalid_support": 0,
            "llm_unclear_count": 0,
            "llm_evidence_span": "",
            "llm_article_outputs": "[]",
        }

    exact_outputs = [o for o in article_outputs if _is_strong_exact(o)]
    nearby_outputs = [o for o in article_outputs if _is_strong_nearby(o)]
    invalid_outputs = [o for o in article_outputs if _is_strong_invalid(o)]
    unclear_outputs = [o for o in article_outputs if str(o.get("location_validation", "") or "") == "unclear"]
    strategic_outputs = [o for o in article_outputs if _is_strategic_supported(o)]
    roi_prior_outputs = [o for o in article_outputs if _is_roi_prior_supported(o, row)]

    actionable = False
    if len(exact_outputs) >= LLM_MIN_EXACT_SUPPORT:
        actionable = True
    elif LLM_ALLOW_SINGLE_ARTICLE_EXACT and len(article_outputs) == 1 and len(exact_outputs) == 1:
        actionable = True
    elif len(exact_outputs) >= 1 and len(nearby_outputs) >= 1:
        actionable = True
    elif LLM_ALLOW_STRATEGIC_SINGLE_SUPPORT and len(strategic_outputs) >= 1:
        actionable = True
    elif LLM_ALLOW_ROI_PRIOR_SINGLE_SUPPORT and len(roi_prior_outputs) >= 1:
        actionable = True

    if actionable:
        if strategic_outputs:
            representative = strategic_outputs[0]
        elif roi_prior_outputs:
            representative = roi_prior_outputs[0]
        elif exact_outputs:
            representative = exact_outputs[0]
        else:
            representative = nearby_outputs[0]
        actionability = "actionable"
        keep = True
        validation_type = str(representative.get("location_validation", "exact_city"))
    elif len(invalid_outputs) >= max(1, len(article_outputs) // 2 + len(article_outputs) % 2) and not exact_outputs and not nearby_outputs:
        representative = invalid_outputs[0]
        actionability = "discard"
        keep = False
        validation_type = str(representative.get("location_validation", "discard"))
    else:
        representative = exact_outputs[0] if exact_outputs else (nearby_outputs[0] if nearby_outputs else article_outputs[0])
        actionability = "suppress"
        keep = True
        validation_type = str(representative.get("location_validation", "unclear"))

    mean_conf = float(np.mean([float(o.get("confidence", 0.0) or 0.0) for o in article_outputs])) if article_outputs else -1.0
    resolved_city = normalize_city_name(representative.get("resolved_location") or row["city"])
    evidence_span = str(representative.get("evidence_span", "") or "").strip()

    return {
        "llm_keep": keep,
        "llm_actionability": actionability,
        "llm_validation_type": validation_type,
        "llm_confidence": mean_conf,
        "llm_reason": str(representative.get("reason", "") or "").strip(),
        "llm_resolved_city": resolved_city,
        "llm_event_summary": str(representative.get("event_summary_ko", "") or "").strip(),
        "llm_imagery_need": str(representative.get("imagery_need_ko", "") or "").strip(),
        "llm_article_count": len(article_outputs),
        "llm_exact_support": len(exact_outputs),
        "llm_nearby_support": len(nearby_outputs),
        "llm_invalid_support": len(invalid_outputs),
        "llm_unclear_count": len(unclear_outputs),
        "llm_strategic_support": len(strategic_outputs),
        "llm_roi_prior_support": len(roi_prior_outputs),
        "llm_evidence_span": evidence_span,
        "llm_article_outputs": json.dumps(article_outputs, ensure_ascii=False),
        "llm_target_category": str(representative.get("target_category", "") or "").strip(),
    }


def verify_top_cities(
    results: pd.DataFrame,
    raw_df: pd.DataFrame,
    url_df: pd.DataFrame | None,
    target_date: str,
    top_n: int = LLM_TOP_N,
) -> pd.DataFrame:
    """상위 후보 도시에 대해 Gemini 기반 위치 검증을 수행한다."""
    if results.empty:
        return results

    verified = _filter_blacklist(results).copy()
    if verified.empty:
        return verified

    # run_daily.py가 요구하는 기본 컬럼 뼈대 채우기
    defaults = {
        "llm_confidence": -1.0, "llm_reason": "", "llm_validation_type": "",
        "llm_resolved_city": "", "llm_event_summary": "", "llm_imagery_need": "",
        "llm_keep": True, "llm_actionability": "unverified", "llm_article_count": 0,
        "llm_exact_support": 0, "llm_nearby_support": 0, "llm_invalid_support": 0,
        "llm_unclear_count": 0, "llm_strategic_support": 0, "llm_evidence_span": "",
        "llm_target_category": "", "llm_article_outputs": "",
    }
    for col, default in defaults.items():
        if col not in verified.columns:
            verified[col] = default
        else:
            verified[col] = verified[col].fillna(default)

    api_key = load_env_key("GEMINI_API_KEY")
    if not api_key:
        print("  [LLM] GEMINI_API_KEY가 없어 검증을 건너뜁니다.")
        return verified

    raw = _prepare_raw_events(raw_df, url_df)
    client = genai.Client(api_key=api_key)
    article_cache: dict[str, Article | None] = {}

    today_mask = verified["date"].astype(str) == str(target_date)
    ranked_today = verified.loc[today_mask].sort_values(["innov_z"], ascending=False).head(top_n)

    # --- 1개 도시를 처리하는 독립 함수 ---
    def process_city(idx, row):
        target_city = row["city"]
        city_norm = normalize_city_key(target_city)
        raw_events = raw.loc[(raw["date"].astype(str) == str(target_date)) & (raw["city_norm"] == city_norm)]
        
        urls, _ = _select_candidate_urls(raw_events)
        articles_data = []

        def _fetch_single_url(url):
            if url not in article_cache:
                article_cache[url] = fetch_article(url)
            return url, article_cache[url]

        # URL 동시 접속 (최대 5개까지만 시도하여 속도 최적화)
        with ThreadPoolExecutor(max_workers=5) as exec_url:
            future_to_url = {exec_url.submit(_fetch_single_url, u): u for u in urls[:5]}
            for future in as_completed(future_to_url):
                u, article = future.result()
                if article and article.body:
                    extracted = extract_relevant_context(article.body, target_city)
                    if extracted:
                        articles_data.append({'text': extracted, 'url': u})
                
                # 핵심 기사를 2개 찾으면 조기 종료 (나머지 무시)
                if len(articles_data) >= 2:
                    break

        if not articles_data:
            return idx, {"llm_actionability": "suppress", "llm_reason": "유효한 언급을 포함한 기사 없음", "llm_validation_type": "TEXT_NOT_FOUND"}

        # 통합 프롬프트로 Gemini에게 1번만 호출
        prompt = build_gemini_prompt(target_city, articles_data, target_date)
        # --- [수정된 부분: 최대 3번까지 끈질기게 재시도] ---
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"response_mime_type": "application/json", "temperature": 0.1}
                )
                llm_result = json.loads(response.text)
                status = llm_result.get("status", "ERROR")
                message = llm_result.get("message", "응답 없음")
                
                actionability = "actionable" if status in ["SUCCESS", "AMBIGUOUS"] else "suppress"
                
                return idx, {
                    "llm_status": status,
                    "llm_actionability": actionability,
                    "llm_reason": message,
                    "llm_event_summary": message if actionability == "actionable" else "",
                    "llm_validation_type": status,
                    "llm_article_count": len(articles_data),
                    "llm_exact_support": len(articles_data) if status == "SUCCESS" else 0
                }
                
            except Exception as e:
                if attempt < 2:  # 에러 나면 2초 대기 후 다시 시도
                    time.sleep(2)
                    continue
                return idx, {"llm_actionability": "suppress", "llm_reason": f"LLM 3회 재시도 실패: {e}", "llm_validation_type": "ERROR"}

    # --- 도시 5개씩 동시에 처리 (Outer Loop 병렬화) ---
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_city, idx, row): idx for idx, row in ranked_today.iterrows()}
        for future in as_completed(futures):
            idx, result_dict = future.result()
            for key, value in result_dict.items():
                verified.loc[idx, key] = value

    return verified.reset_index(drop=True)


# --- [추가] Track 2: 정기 관측 도시(Baseline) 전황 요약 프롬프트 ---
def build_baseline_prompt(target_city: str, articles_data: list, target_date: str) -> str:
    articles_context = ""
    for idx, data in enumerate(articles_data, 1):
        articles_context += f"\n[Article {idx}]\nContent: {data['text']}\n"

    return f"""
    [Task] Summarize the macro military/political situation for [{target_city}] on {target_date}.
    [Context] This city is a known conflict zone. Read the provided text chunks.
    
    [Input]
    {articles_context}
    
    [Instruction]
    Write exactly ONE sentence in Korean summarizing the overall war situation, military tension, or political conflict happening in or around this city. 
    Focus on macro trends (e.g., heightened alert, ongoing skirmishes, political shifts) rather than individual isolated incidents.
    If there is not enough information, respond exactly with "특이 동향이 명확히 포착되지 않았습니다."
    """

# --- [추가] Track 2: 정기 관측 도시 요약 실행 함수 ---
def summarize_baseline_cities(baseline_df: pd.DataFrame, raw_df: pd.DataFrame, url_df: pd.DataFrame, target_date: str, top_n: int = 5) -> pd.DataFrame:
    """Track 2 전략 도시들에 대해 1줄 전황 요약을 생성합니다."""
    if baseline_df.empty:
        return baseline_df

    baseline_df = baseline_df.copy()
    baseline_df['llm_baseline_summary'] = "요약 대기"
    
    api_key = load_env_key("GEMINI_API_KEY")
    if not api_key: return baseline_df
    
    client = genai.Client(api_key=api_key)
    raw = _prepare_raw_events(raw_df, url_df)
    article_cache = {}

    def process_baseline(idx, row):
        target_city = row["city"]
        city_norm = normalize_city_key(target_city)
        raw_events = raw.loc[(raw["date"].astype(str) == str(target_date)) & (raw["city_norm"] == city_norm)]
        urls, _ = _select_candidate_urls(raw_events)
        
        articles_data = []
        for url in urls[:3]: # 전황 파악은 상위 3개 기사만 가볍게 확인
            if url not in article_cache:
                article_cache[url] = fetch_article(url)
            article = article_cache[url]
            if article and article.body:
                extracted = extract_relevant_context(article.body, target_city)
                if extracted:
                    articles_data.append({'text': extracted, 'url': url})
        
        if not articles_data:
            return idx, "관련 기사 확보 실패"

        prompt = build_baseline_prompt(target_city, articles_data, target_date)
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"temperature": 0.1}
            )
            return idx, response.text.strip()
        except Exception:
            return idx, "요약 생성 실패"

    # 병렬 처리로 요약
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_baseline, idx, row): idx for idx, row in baseline_df.head(top_n).iterrows()}
        for future in as_completed(futures):
            idx, summary = future.result()
            baseline_df.loc[idx, 'llm_baseline_summary'] = summary

    return baseline_df