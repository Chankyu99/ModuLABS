from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
import trafilatura
from bs4 import BeautifulSoup
from google import genai
from newspaper import Article as NewspaperArticle


MODEL_CANDIDATES = [
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash",
]

HIGH_VALUE_CODES = {194, 195, 1951, 1952}
MONITORED_COUNTRIES = {"IRN", "ISR", "USA", "LBN", "IRQ", "SYR", "PSE", "YEM", "SAU", "ARE", "QAT", "KWT"}
TARGET_TYPES = {
    "airport",
    "airbase",
    "power_plant",
    "substation",
    "refinery",
    "gas_facility",
    "port",
    "bridge",
    "rail_hub",
    "dam",
    "warehouse",
    "missile_site",
    "radar_site",
    "military_base",
    "other_infrastructure",
}
COUNTRY_ONLY_NAMES = {
    "iran",
    "iraq",
    "israel",
    "lebanon",
    "syria",
    "yemen",
    "saudi arabia",
    "united arab emirates",
    "uae",
    "qatar",
    "kuwait",
    "bahrain",
    "oman",
    "jordan",
    "egypt",
    "turkey",
    "middle east",
}

REQUEST_TIMEOUT = 20
MAX_BODY_CHARS = 4000
MAX_EVENTS = 12
MAX_ARTICLES_PER_EVENT = 2


SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "is_attack",
        "actual_location",
        "location_granularity",
        "target_type",
        "is_fixed_infrastructure",
        "verification_need",
        "reason",
    ],
    "properties": {
        "is_attack": {"type": "boolean"},
        "actual_location": {"type": "string"},
        "location_granularity": {
            "type": "string",
            "enum": ["city", "country", "unknown"],
        },
        "target_type": {
            "type": "string",
            "enum": sorted(TARGET_TYPES | {"unknown", "non_infrastructure"}),
        },
        "is_fixed_infrastructure": {"type": "boolean"},
        "verification_need": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "reason": {"type": "string"},
    },
}

SYSTEM_PROMPT = """You are a simple classifier for post-strike satellite tasking.

Read a news article bundle and extract only:
- whether this is a real attack event,
- the actual attacked location,
- whether the location is city-level or only country-level,
- whether the target is fixed infrastructure,
- the target type,
- whether verification need is high, medium, or low.

Rules:
- Prefer the attacked location over actor home country.
- If the article only names a country, set location_granularity to country.
- If the article does not clearly identify a city, do not invent one.
- Return JSON only.
"""


@dataclass
class Article:
    url: str
    title: str
    source: str
    published_utc: str | None
    body: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYYMMDD")
    return parser.parse_args()


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
        "User-Agent": "Mozilla/5.0 (compatible; SIA-Simple-Test/1.0)",
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


def load_daily_data(target_date: str) -> pd.DataFrame:
    daily_parquet = Path("data/daily") / f"{target_date}.parquet"
    if not daily_parquet.exists():
        raise FileNotFoundError(f"missing daily parquet: {daily_parquet}")

    df = pd.read_parquet(daily_parquet)
    for col in ["EventCode", "NumSources", "NumMentions", "NumArticles", "AvgTone"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prefilter_events(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["EventCode"].isin(HIGH_VALUE_CODES)
        & (
            df["Actor1CountryCode"].isin(MONITORED_COUNTRIES)
            | df["Actor2CountryCode"].isin(MONITORED_COUNTRIES)
        )
        & (df["ActionGeo_Type"].astype(str) == "4")
    )
    filtered = df.loc[mask].copy()
    filtered["candidate_weight"] = (
        filtered["NumSources"].fillna(0) * 4
        + filtered["NumMentions"].fillna(0)
        + filtered["NumArticles"].fillna(0)
    )
    return filtered.sort_values(
        ["candidate_weight", "NumSources", "NumMentions"],
        ascending=[False, False, False],
    ).head(MAX_EVENTS)


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


def call_gemini(client: genai.Client, event_meta: dict[str, Any], articles: list[Article]) -> dict[str, Any]:
    prompt = f"""Analyze this event and article bundle.

Event metadata:
- EventCode: {event_meta.get('EventCode')}
- GDELT ActionGeo_FullName: {event_meta.get('ActionGeo_FullName')}
- Actor1CountryCode: {event_meta.get('Actor1CountryCode')}
- Actor2CountryCode: {event_meta.get('Actor2CountryCode')}
- NumSources: {event_meta.get('NumSources')}

Article bundle:
{build_article_bundle(articles)}
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
                },
            )
            return json.loads(response.text)
        except Exception as exc:
            last_error = exc
            print(f"[llm] model failed: {model_name} -> {exc}", flush=True)
            time.sleep(1)

    if last_error is None:
        raise RuntimeError("all models failed")
    raise last_error


def is_city_level_location(location_name: str | None, granularity: str | None) -> bool:
    if granularity != "city":
        return False
    normalized = clean_text(location_name or "").lower()
    if not normalized:
        return False
    if normalized in COUNTRY_ONLY_NAMES:
        return False
    return True


def simple_score(event_meta: dict[str, Any], llm_output: dict[str, Any]) -> int:
    score = 0
    if llm_output["is_attack"]:
        score += 1
    if is_city_level_location(llm_output["actual_location"], llm_output["location_granularity"]):
        score += 1
    if llm_output["is_fixed_infrastructure"]:
        score += 1
    if int(float(event_meta.get("NumSources", 0) or 0)) >= 20:
        score += 1
    if llm_output["verification_need"] in {"high", "medium"}:
        score += 1
    return score


def classify_event(event_meta: dict[str, Any], llm_output: dict[str, Any]) -> str:
    if not llm_output["is_attack"]:
        return "discard"
    if not llm_output["is_fixed_infrastructure"]:
        return "discard"
    if llm_output["target_type"] not in TARGET_TYPES:
        return "discard"
    if not is_city_level_location(llm_output["actual_location"], llm_output["location_granularity"]):
        return "hold"

    score = simple_score(event_meta, llm_output)
    if score >= 4:
        return "request_imagery"
    if score >= 3:
        return "hold"
    return "discard"


def run_for_date(target_date: str) -> dict[str, Any]:
    api_key = load_env_key("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")

    client = genai.Client(api_key=api_key)
    df = load_daily_data(target_date)
    candidates = prefilter_events(df)

    print(f"[simple] target date: {target_date}", flush=True)
    print(f"[simple] daily rows: {len(df):,}", flush=True)
    print(f"[simple] evaluating top events: {len(candidates):,}", flush=True)

    records = []
    for _, row in candidates.iterrows():
        event_meta = row.to_dict()
        urls = []
        if isinstance(event_meta.get("SOURCEURL"), str) and event_meta["SOURCEURL"]:
            urls.append(event_meta["SOURCEURL"])
        urls = urls[:MAX_ARTICLES_PER_EVENT]

        articles = []
        for url in urls:
            article = fetch_article(url)
            if article:
                articles.append(article)

        if not articles:
            print(f"[skip] {event_meta['GLOBALEVENTID']} no article content", flush=True)
            continue

        try:
            llm_output = call_gemini(client, event_meta, articles)
        except Exception as exc:
            print(f"[skip] {event_meta['GLOBALEVENTID']} llm failed: {exc}", flush=True)
            continue

        score = simple_score(event_meta, llm_output)
        decision = classify_event(event_meta, llm_output)
        record = {
            "event_id": str(event_meta["GLOBALEVENTID"]),
            "gdelt_city": event_meta.get("ActionGeo_FullName"),
            "resolved_city": llm_output["actual_location"],
            "location_granularity": llm_output["location_granularity"],
            "event_code": int(event_meta["EventCode"]) if pd.notna(event_meta["EventCode"]) else None,
            "num_sources": int(event_meta["NumSources"]) if pd.notna(event_meta["NumSources"]) else None,
            "decision": decision,
            "simple_score": score,
            "llm_output": llm_output,
            "source_url": event_meta.get("SOURCEURL"),
        }
        records.append(record)
        print(
            f"[event] {event_meta['ActionGeo_FullName']:15s} "
            f"id={event_meta['GLOBALEVENTID']} decision={decision} "
            f"resolved={llm_output['actual_location']} granularity={llm_output['location_granularity']} "
            f"target={llm_output['target_type']} score={score}",
            flush=True,
        )

    selected = [r for r in records if r["decision"] in {"request_imagery", "hold"} and r["location_granularity"] == "city"]
    by_city: dict[str, list[dict[str, Any]]] = {}
    for record in selected:
        by_city.setdefault(record["resolved_city"], []).append(record)

    city_summary = []
    for city, items in by_city.items():
        items = sorted(items, key=lambda x: (x["decision"] != "request_imagery", -x["simple_score"]))
        city_summary.append(
            {
                "city": city,
                "event_count": len(items),
                "events": [
                    {
                        "event_id": item["event_id"],
                        "decision": item["decision"],
                        "simple_score": item["simple_score"],
                        "gdelt_city": item["gdelt_city"],
                        "target_type": item["llm_output"]["target_type"],
                        "verification_need": item["llm_output"]["verification_need"],
                    }
                    for item in items
                ],
            }
        )

    result = {
        "target_date": target_date,
        "evaluated_events": len(records),
        "request_imagery_events": sum(1 for r in records if r["decision"] == "request_imagery"),
        "hold_events": sum(1 for r in records if r["decision"] == "hold"),
        "discard_events": sum(1 for r in records if r["decision"] == "discard"),
        "selected_cities": city_summary,
        "records": records,
    }

    out_path = Path("output") / f"{target_date}_simple_city_selector.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[simple] saved: {out_path}", flush=True)
    return result


def main() -> None:
    args = parse_args()
    run_for_date(args.date)


if __name__ == "__main__":
    main()
