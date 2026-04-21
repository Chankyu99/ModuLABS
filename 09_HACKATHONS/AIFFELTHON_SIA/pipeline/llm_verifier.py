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
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import threading
import pandas as pd
import requests
import trafilatura
from bs4 import BeautifulSoup
from google import genai
from newspaper import Article as NewspaperArticle
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline.city_utils import normalize_city_key, normalize_city_name
from pipeline.config import (
    CITY_BLACKLIST,
    LLM_PREFETCH_URLS,
    LLM_TOP_K_URLS,
    LLM_TOP_N,
)


REQUEST_TIMEOUT = 4
_cache_lock = threading.Lock()
MAX_BODY_CHARS = 4000
MAX_ARTICLES_PER_CANDIDATE = max(1, LLM_TOP_K_URLS)
MAX_PREFETCH_URLS = max(MAX_ARTICLES_PER_CANDIDATE, LLM_PREFETCH_URLS)


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

# Legacy helper disabled: used by the old structured per-article verifier.
# def build_article_bundle(articles: list[Article]) -> str:
#     ...


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

    return f"""
    [Task] Evaluate military/conflict status for [{target_city}] on {target_date}.
    [Context] Read the provided text chunks. Ignore external metadata. Datelines (reporter locations) are not conflict zones.

    [Input]
    {articles_context}

    [Definitions for Status]
    SUCCESS: Direct military attack or physical conflict explicitly occurred IN or ON [{target_city}].
    AMBIGUOUS: [{target_city}] is under indirect military tension (e.g., air raid sirens, evacuations, staging area) without direct strikes.
    DATE_MISMATCH: The event matches the location, but clearly occurred in the past (e.g., 'last year', 'last June').
    DROPPED: [{target_city}] is merely a dateline, not the target, or the text lacks evidence.

    [Instructions for Message]
    If Status is SUCCESS or AMBIGUOUS: Write a 1-line summary specifying WHO did WHAT to WHOM in Korean.
    If Status is DATE_MISMATCH or DROPPED: Write a 1-line reason explaining why it was rejected in Korean.
    Return the status token exactly as one of: SUCCESS, AMBIGUOUS, DATE_MISMATCH, DROPPED.

    [JSON Schema]
    {{
        "status": "SUCCESS" | "AMBIGUOUS" | "DATE_MISMATCH" | "DROPPED",
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

    meta = {}
    # URL slug 자체에 도시명이 있으면 약간 우선순위를 준다.
    urls = sorted(
        urls,
        key=lambda url: (
            -sum(1 for variant in _candidate_city_variants(candidate_city) if variant and variant.replace(" ", "-") in url.lower()),
            urls.index(url),
        ),
    )

    return urls, meta



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
            with _cache_lock:
                if url in article_cache:
                    return url, article_cache[url]
            article = fetch_article(url)
            with _cache_lock:
                article_cache[url] = article
            return url, article

        # URL 동시 접속 — 2개 확보 즉시 나머지 취소
        exec_url = ThreadPoolExecutor(max_workers=5)
        try:
            future_to_url = {exec_url.submit(_fetch_single_url, u): u for u in urls[:5]}
            for future in as_completed(future_to_url):
                u, article = future.result()
                if article and article.body:
                    extracted = extract_relevant_context(article.body, target_city)
                    if extracted:
                        articles_data.append({'text': extracted, 'url': u})
                if len(articles_data) >= 2:
                    break
        finally:
            exec_url.shutdown(wait=False, cancel_futures=True)

        if not articles_data:
            return idx, {"llm_actionability": "suppress", "llm_reason": "유효한 언급을 포함한 기사 없음", "llm_validation_type": "TEXT_NOT_FOUND"}

        # 통합 프롬프트로 Gemini에게 1번만 호출
        prompt = build_gemini_prompt(target_city, articles_data, target_date)
        # 503/UNAVAILABLE 같은 재시도 가치가 있는 에러만 지수 백오프로 재시도.
        # 고정 2초 대신 1→2→4→8→16초 + 랜덤 jitter로 서버 과부하 회복 시간 확보.
        RETRYABLE_MARKERS = ("503", "UNAVAILABLE", "overloaded", "429", "RESOURCE_EXHAUSTED")
        for attempt in range(5):
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
                err_text = str(e)
                is_retryable = any(marker in err_text for marker in RETRYABLE_MARKERS)
                if is_retryable and attempt < 4:
                    delay = (2 ** attempt) + random.uniform(0, 1)  # 1→2→4→8→16초 + jitter
                    time.sleep(delay)
                    continue
                return idx, {"llm_actionability": "suppress", "llm_reason": f"LLM {attempt+1}회 재시도 실패: {e}", "llm_validation_type": "ERROR"}

    # --- 동시 호출을 2개로 제한: Gemini 서버 부담 완화 ---
    with ThreadPoolExecutor(max_workers=2) as executor:
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
        for url in urls[:3]:
            with _cache_lock:
                cached = article_cache.get(url)
            if cached is None:
                cached = fetch_article(url)
                with _cache_lock:
                    article_cache[url] = cached
            if cached and cached.body:
                extracted = extract_relevant_context(cached.body, target_city)
                if extracted:
                    articles_data.append({'text': extracted, 'url': url})
        
        if not articles_data:
            return idx, "관련 기사 확보 실패"

        prompt = build_baseline_prompt(target_city, articles_data, target_date)
        RETRYABLE_MARKERS = ("503", "UNAVAILABLE", "overloaded", "429", "RESOURCE_EXHAUSTED")
        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"temperature": 0.1}
                )
                return idx, response.text.strip()
            except Exception as e:
                err_text = str(e)
                is_retryable = any(marker in err_text for marker in RETRYABLE_MARKERS)
                if is_retryable and attempt < 4:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                return idx, "요약 생성 실패"
        return idx, "요약 생성 실패"

    # 동시 호출을 2개로 제한 (이전 5)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_baseline, idx, row): idx for idx, row in baseline_df.head(top_n).iterrows()}
        for future in as_completed(futures):
            idx, summary = future.result()
            baseline_df.loc[idx, 'llm_baseline_summary'] = summary

    return baseline_df
