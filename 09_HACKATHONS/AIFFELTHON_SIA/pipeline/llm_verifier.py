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


REQUEST_TIMEOUT = 20
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

    # NaN이 그대로 출력되지 않도록 기본 필드를 먼저 채운다.
    defaults = {
        "llm_confidence": -1.0,
        "llm_reason": "",
        "llm_validation_type": "",
        "llm_resolved_city": "",
        "llm_event_summary": "",
        "llm_imagery_need": "",
        "llm_keep": True,
        "llm_actionability": "unverified",
        "llm_article_count": 0,
        "llm_exact_support": 0,
        "llm_nearby_support": 0,
        "llm_invalid_support": 0,
        "llm_unclear_count": 0,
        "llm_strategic_support": 0,
        "llm_evidence_span": "",
        "llm_target_category": "",
        "llm_article_outputs": "",
    }
    for col, default in defaults.items():
        if col not in verified.columns:
            verified[col] = default
        else:
            verified[col] = verified[col].fillna(default)

    api_key = load_env_key("GEMINI_API_KEY")
    if not api_key:
        print("  [LLM] GEMINI_API_KEY가 없어 LLM 검증을 건너뜁니다.")
        return verified

    raw = _prepare_raw_events(raw_df, url_df)
    article_cache: dict[str, Article | None] = {}

    today_mask = verified["date"].astype(str) == str(target_date)
    if not today_mask.any():
        return verified

    ranked_today = (
        verified.loc[today_mask]
        .sort_values(["risk_level", "innov_z", "conflict_index"], ascending=[False, False, False])
        .head(top_n)
        .copy()
    )

    client = genai.Client(api_key=api_key)
    drop_indices = []

    for idx, row in ranked_today.iterrows():
        city_norm = normalize_city_key(row["city"])
        raw_events = raw.loc[
            (raw["date"].astype(str) == str(target_date))
            & (raw["city_norm"] == city_norm)
        ].copy()

        urls, raw_meta = _select_candidate_urls(raw_events)
        articles: list[Article] = []

        def _fetch_single_url(url):
            if url not in article_cache:
                return url, fetch_article(url)
            return url, article_cache[url]

        # max_workers=10 은 10개의 웹사이트에 동시에 접속한다는 뜻입니다.
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(_fetch_single_url, u): u for u in urls}
            for future in as_completed(future_to_url):
                u, article = future.result()
                article_cache[u] = article
                if article is not None:
                    articles.append(article)

        if not articles:
            agg = _aggregate_article_outputs([], row)
            for key, value in agg.items():
                verified.loc[idx, key] = value
            continue
        
        # 병렬 처리 수정 전 코드
        # for url in urls:
        #     if url not in article_cache:
        #         article_cache[url] = fetch_article(url)
        #     article = article_cache[url]
        #     if article is not None:
        #         articles.append(article)

        # if not articles:
        #     agg = _aggregate_article_outputs([], row)
        #     for key, value in agg.items():
        #         verified.loc[idx, key] = value
        #     continue

        articles = sorted(
            articles,
            key=lambda article: (
                _article_city_score(article, str(row["city"])),
                _article_target_score(article),
            ),
            reverse=True,
        )[:MAX_ARTICLES_PER_CANDIDATE]

        candidate_meta = {
            "city": row["city"],
            "date": row["date"],
            "conflict_index": row.get("conflict_index"),
            "innov_z": row.get("innov_z"),
            "events": row.get("events"),
            "lat": row.get("lat"),
            "lon": row.get("lon"),
            **raw_meta,
        }

        article_outputs: list[dict[str, Any]] = []
        article_errors: list[str] = []
        
        def _call_llm_for_article(article):
            try:
                out = call_gemini(client, candidate_meta, article)
                out["source_url"] = article.url
                out["source_name"] = article.source
                return out
            except Exception as exc:
                return f"{article.url}: {exc}"

        # max_workers=3 은 Gemini API에 3개의 질문을 동시에 던진다는 뜻입니다.
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_article = {executor.submit(_call_llm_for_article, art): art for art in articles}
            for future in as_completed(future_to_article):
                res = future.result()
                if isinstance(res, dict):
                    article_outputs.append(res)
                else:
                    article_errors.append(res)
        
        # 병렬 처리 수정 전 코드            
        # for article in articles:
        #     try:
        #         article_output = call_gemini(client, candidate_meta, article)
        #         article_output["source_url"] = article.url
        #         article_output["source_name"] = article.source
        #         article_outputs.append(article_output)
        #     except Exception as exc:
        #         article_errors.append(f"{article.url}: {exc}")

        if not article_outputs:
            agg = _aggregate_article_outputs([], row)
            agg["llm_reason"] = (
                "기사는 확보했지만 LLM 기사별 검증이 실패해 원본 후보를 유지했습니다. "
                + " | ".join(article_errors[:2])
            ).strip()
            agg["llm_validation_type"] = "llm_error"
            agg["llm_actionability"] = "suppress"
            for key, value in agg.items():
                verified.loc[idx, key] = value
            continue

        agg = _aggregate_article_outputs(article_outputs, row)
        if article_errors and not agg["llm_reason"]:
            agg["llm_reason"] = "일부 기사 검증 실패가 있었으나 나머지 기사 기준으로 판단했습니다."
        for key, value in agg.items():
            verified.loc[idx, key] = value

        if not agg["llm_keep"]:
            drop_indices.append(idx)

    if drop_indices:
        verified = verified.drop(index=drop_indices)

    for col, default in defaults.items():
        verified[col] = verified[col].fillna(default)

    return verified.reset_index(drop=True)
