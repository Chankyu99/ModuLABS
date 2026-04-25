import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

import pandas as pd
import trafilatura
from trafilatura.metadata import extract_metadata
from google import genai
from google.genai import types

from src.config import (
    API_KEY,
    LLM_MODELS
)
from src import cache

# ── Parallelism caps (tune if rate-limited) ──────────────────────────────
CITY_CONCURRENCY   = 3    # cities processed in parallel (≤ Gemini QPS budget)
SCRAPE_CONCURRENCY = 6    # URL fetches in parallel per city
# Global max concurrent HTTP = CITY_CONCURRENCY × SCRAPE_CONCURRENCY ≈ 18

client = genai.Client(api_key=API_KEY)
MODEL_ID          = LLM_MODELS[1]   # gemini-2.5-flash (primary)
FALLBACK_MODEL_ID = LLM_MODELS[0]   # gemini-2.5-flash-lite (less congested, used on primary exhaustion)


def scrape_article(url: str) -> tuple:
    """Returns (status, text). status ∈ {'ok', 'unreachable', 'no_text'}; text may be None."""
    hit = cache.get_scrape(url)
    if hit is not None:
        return hit["status"], hit["text"]

    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            print(f"  [DEBUG] 사이트 접근 차단/응답 없음: {url}")
            cache.set_scrape(url, "unreachable", None)
            return "unreachable", None
        text = trafilatura.extract(downloaded, include_links=False, include_images=False)
        if text:
            cache.set_scrape(url, "ok", text)
            return "ok", text
        cache.set_scrape(url, "no_text", None)
        return "no_text", None
    except Exception as e:
        print(f"  [DEBUG] 스크래핑 오류 발생: {url} ({e})")
        cache.set_scrape(url, "unreachable", None)
        return "unreachable", None
    
def extract_relevant_context(text: str, target_city: str) -> str:
    target_lower = target_city.lower()
    if text.lower().count(target_lower) < 1:
        return None

    # 문장 단위 분할 (문단 구조에 의존하지 않음)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return None

    # 1) 리드: 기사 첫 3문장 (날짜 앵커 확보용)
    lead_indices = set(range(min(3, len(sentences))))

    # 2) 지명 언급 ± 앞뒤 2문장 (상위 2개 언급)
    found_indices = [i for i, s in enumerate(sentences) if target_lower in s.lower()]
    context_indices = set()
    for idx in found_indices[:2]:
        context_indices.update(range(max(0, idx - 2), min(len(sentences), idx + 3)))

    if not context_indices:
        return None

    # 3) 리드와 본문 언급 병합, 원래 순서 유지. 사이 간격이 있으면 [...] 마커 삽입
    all_indices = sorted(lead_indices | context_indices)
    chunks = []
    prev = None
    for i in all_indices:
        if prev is not None and i > prev + 1:
            chunks.append("[...]")
        chunks.append(sentences[i])
        prev = i
    return " ".join(chunks)


PROMPT_VERSION = "v6"

def build_gemini_prompt(target_city: str, articles_data: list, target_date: str) -> str:
    articles_context = ""
    for idx, data in enumerate(articles_data, 1):
        articles_context += f"\n[Article {idx}]\nContent: {data['text']}\n"

    prompt = f"""City: {target_city}
Target date: {target_date} (YYYYMMDD)

Question: Did a physical conflict event (attack, strike, bombing, fighting, sirens, evacuation, troop movement) occur in or very near {target_city} within the past 7 days of the target date?

Rules:
- "Yesterday", "last night", "hours ago", "this morning", day-of-week references (e.g. "Monday", "Saturday") → inside window.
- Old years (1979, 2024), "last week/month/year", anniversary framing → out of window.
- "Last week" is borderline — use DATE_MISMATCH only if clearly beyond 7 days.

Labels:
- SUCCESS: direct attack/strike/bombing/fighting in {target_city} within the window.
- AMBIGUOUS: only indirect tension (sirens, evacuation, troop movement, staging) within the window.
- DATE_MISMATCH: conflict event exists but clearly out of window (more than 7 days ago).
- DROPPED: {target_city} appears only as a dateline, or no conflict content about it.
- NO_MENTION: {target_city} not meaningfully mentioned in any article.

Output JSON only:
{{"status": "SUCCESS" | "AMBIGUOUS" | "DATE_MISMATCH" | "DROPPED" | "NO_MENTION", "message": "<one line: attacker / type / target+impact, or reason>"}}

Articles:
{articles_context}
"""
    return prompt

GEMINI_CALL_DELAY_SEC = 1.5   # baseline pacing before each call
GEMINI_RETRY_WAIT_SEC = 2.5   # initial wait on transient failure (backoff multiplies this)
GEMINI_MAX_RETRIES    = 4     # per-model attempts = 1 + GEMINI_MAX_RETRIES
GEMINI_CALL_TIMEOUT   = 30    # seconds per single API call (beyond this = transient timeout)
GEMINI_FINAL_WAIT_SEC = 60    # after both models exhausted, wait this long then try primary one more time

# Substrings that identify a transient failure worth retrying.
# Covers: rate limits, service outages, and ALL network/protocol errors observed
# (RemoteProtocolError "Server disconnected", ConnectError DNS failures, timeouts).
_TRANSIENT_MARKERS = (
    '503', 'UNAVAILABLE', '429', 'RESOURCE_EXHAUSTED', '500', 'INTERNAL',
    '504', 'DEADLINE_EXCEEDED', 'timeout', 'Timeout',
    'Server disconnected', 'RemoteProtocolError', 'ConnectError',
    'Connection', 'nodename nor servname', 'Temporary failure',
    'EOF occurred', 'SSL',
)


def _is_transient(exc: Exception) -> bool:
    name = type(exc).__name__
    msg = str(exc)
    return any(m in msg or m in name for m in _TRANSIENT_MARKERS)


def _invoke_model(model_id: str, prompt: str):
    return client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            safety_settings=[
                types.SafetySetting(category=c, threshold='BLOCK_NONE')
                for c in [
                    'HARM_CATEGORY_HARASSMENT', 'HARM_CATEGORY_HATE_SPEECH',
                    'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'HARM_CATEGORY_DANGEROUS_CONTENT'
                ]
            ]
        )
    )


def _try_model_with_retries(model_id: str, prompt: str, label: str):
    """Returns (response, exhausted_transient). exhausted_transient=True means all retries failed on
    transient errors — worth trying another model or waiting. None means non-transient giveup."""
    for attempt in range(1 + GEMINI_MAX_RETRIES):
        try:
            with ThreadPoolExecutor(max_workers=1) as _tp:
                fut = _tp.submit(_invoke_model, model_id, prompt)
                return fut.result(timeout=GEMINI_CALL_TIMEOUT), False
        except FuturesTimeoutError:
            if attempt == GEMINI_MAX_RETRIES:
                print(f"  [GEMINI {label} EXHAUSTED] timeout after {GEMINI_MAX_RETRIES} retries")
                return None, True
            wait = GEMINI_RETRY_WAIT_SEC + attempt * 0.5
            print(f"  [GEMINI {label} RETRY {attempt+1}/{GEMINI_MAX_RETRIES}] TimeoutError ({GEMINI_CALL_TIMEOUT}s); sleeping {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            if not _is_transient(e):
                print(f"  [GEMINI {label} NON-TRANSIENT, giving up] {type(e).__name__}: {e}")
                return None, False
            if attempt == GEMINI_MAX_RETRIES:
                print(f"  [GEMINI {label} EXHAUSTED] {type(e).__name__}: {str(e)[:150]}")
                return None, True
            wait = GEMINI_RETRY_WAIT_SEC + attempt * 0.5
            print(f"  [GEMINI {label} RETRY {attempt+1}/{GEMINI_MAX_RETRIES}] {type(e).__name__}; sleeping {wait:.1f}s")
            time.sleep(wait)
    return None, True


def call_gemini_verification(prompt: str) -> dict:
    """
    Robust call: primary → fallback model → (sleep 60s) → primary once more.
    Returns None only if all three stages fail or a non-transient error occurs.
    """
    time.sleep(GEMINI_CALL_DELAY_SEC)

    # Stage 1: primary model
    response, exhausted = _try_model_with_retries(MODEL_ID, prompt, "PRIMARY")
    if response is not None:
        pass
    elif not exhausted:
        return None  # non-transient giveup
    else:
        # Stage 2: fallback model (lite is typically less congested)
        print(f"  [GEMINI FALLBACK] switching to {FALLBACK_MODEL_ID}")
        response, exhausted2 = _try_model_with_retries(FALLBACK_MODEL_ID, prompt, "FALLBACK")
        if response is None and exhausted2:
            # Stage 3: last-ditch long wait then primary once more
            print(f"  [GEMINI COOLDOWN] sleeping {GEMINI_FINAL_WAIT_SEC}s then retrying PRIMARY")
            time.sleep(GEMINI_FINAL_WAIT_SEC)
            response, _ = _try_model_with_retries(MODEL_ID, prompt, "FINAL")
            if response is None:
                print("  [GEMINI GIVE UP] all strategies exhausted — row will be ERROR")
                return None
        elif response is None:
            return None  # non-transient in fallback

    # Surface safety / finish-reason / empty-response issues
    raw_text = getattr(response, "text", None) or ""
    if not raw_text:
        try:
            fin = response.candidates[0].finish_reason if response.candidates else None
            sr  = response.candidates[0].safety_ratings if response.candidates else None
            print(f"  [GEMINI ERROR: empty response] finish_reason={fin} safety={sr}")
        except Exception as e:
            print(f"  [GEMINI ERROR: empty response, inspection failed] {e}")
        return None

    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if not json_match:
        print(f"  [GEMINI ERROR: no JSON in response] preview={raw_text[:200]!r}")
        return None

    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError as e:
        print(f"  [GEMINI ERROR: JSON parse] {e} — preview={json_match.group()[:200]!r}")
        return None

#
# llm_status values (set per anomaly row):
#   UNVERIFIED         — not evaluated (outside top-K)
#   ARTICLE_UNREACHABLE — all URLs failed to fetch or returned no text (exclude from precision denom)
#   NO_MENTION         — URLs fetched OK but none mentioned the target city (FP: filter lead bad)
#   SUCCESS            — LLM confirmed direct conflict (TP)
#   AMBIGUOUS          — LLM confirmed indirect tension (TP)
#   DATE_MISMATCH      — LLM found event on different date (FP)
#   DROPPED            — LLM rejected (FP)
#   ERROR              — LLM API failure (exclude from precision denom)
#
def _scrape_and_extract(url: str, target_city: str) -> tuple:
    """Thread worker: fetch + mention-context extract. Returns (url, status, extracted_or_None, mention_count)."""
    status, text = scrape_article(url)
    if status != 'ok' or not text:
        return (url, status, None, 0)
    mention_count = text.lower().count(target_city.lower())
    extracted = extract_relevant_context(text, target_city)
    return (url, status, extracted, mention_count)


def _process_city(row: pd.Series, filtered_df: pd.DataFrame, url_df: pd.DataFrame, target_date: str) -> dict:
    """Run one city's scrape+LLM pipeline. Returns result dict for main-thread writeback."""
    target_city = row['city']
    feature_id  = row['ActionGeo_FeatureID']
    logs = [f"\nTarget: {target_city} | Z-Score: {row['innov_z']:.1f}"]

    city_events = filtered_df[(filtered_df['ActionGeo_FeatureID'] == feature_id) & (filtered_df['date'] == target_date)]
    potential_events = city_events.nlargest(30, 'weighted_index')
    merged_events = potential_events.merge(url_df[['GLOBALEVENTID', 'SOURCEURL']], on='GLOBALEVENTID', how='inner')
    unique_urls = merged_events['SOURCEURL'].unique().tolist()

    scrape_stats = {'ok': 0, 'unreachable': 0, 'no_text': 0, 'no_mention': 0, 'matched': 0}
    logs.append(f"  -> Scanning {len(unique_urls)} sources in parallel for '{target_city}'...")

    # Parallel scrape + extract. Determinism: submit all, then iterate in original order.
    results_by_url = {}
    if unique_urls:
        with ThreadPoolExecutor(max_workers=SCRAPE_CONCURRENCY) as ex:
            futs = {ex.submit(_scrape_and_extract, u, target_city): u for u in unique_urls}
            for fut in as_completed(futs):
                try:
                    url, status, extracted, mc = fut.result()
                    results_by_url[url] = (status, extracted, mc)
                except Exception as e:
                    results_by_url[futs[fut]] = ('unreachable', None, 0)
                    logs.append(f"     [ERROR] scrape future failed: {e}")

    # Collect first 3 matches in original URL order (deterministic)
    articles_data, valid_urls = [], []
    for url in unique_urls:
        status, extracted, mc = results_by_url.get(url, ('unreachable', None, 0))
        if status == 'ok':
            scrape_stats['ok'] += 1
            if extracted:
                scrape_stats['matched'] += 1
                if len(articles_data) < 3:
                    articles_data.append({'text': extracted})
                    valid_urls.append(url)
                    logs.append(f"     [MATCH] ({mc} mentions): {url[:60]}...")
            else:
                scrape_stats['no_mention'] += 1
        else:
            scrape_stats[status] = scrape_stats.get(status, 0) + 1

    result = {
        'source_urls':  valid_urls,
        'scrape_stats': json.dumps(scrape_stats),
        'llm_status':   None,
        'llm_report':   None,
        'is_anomaly':   True,
        'logs':         logs,
    }

    # No valid articles → separate failure causes
    if not articles_data:
        if scrape_stats['ok'] == 0 and (scrape_stats['unreachable'] + scrape_stats['no_text']) > 0:
            final_status = 'ARTICLE_UNREACHABLE'
            msg = f"모든 URL 접근 불가/본문 없음 (unreachable={scrape_stats['unreachable']}, no_text={scrape_stats['no_text']})"
        else:
            final_status = 'NO_MENTION'
            msg = f"기사는 받았으나 '{target_city}' 언급 없음 (ok={scrape_stats['ok']}, no_mention={scrape_stats['no_mention']})"
        result['llm_status'] = final_status
        result['llm_report'] = json.dumps({"Summary": msg}, ensure_ascii=False)
        result['is_anomaly'] = False
        return result

    # LLM call with cache
    article_texts = [d['text'] for d in articles_data]
    cache_key = cache.make_llm_key(PROMPT_VERSION, target_city, target_date, article_texts)
    llm_result = cache.get_llm(cache_key)
    if llm_result is not None:
        logs.append(f"  -> [CACHE HIT] Gemini response reused for '{target_city}'")
    else:
        logs.append(f"  -> Sending {len(articles_data)} articles to Gemini for '{target_city}'...")
        try:
            llm_result = call_gemini_verification(build_gemini_prompt(target_city, articles_data, target_date))
        except Exception as e:
            logs.append(f"  -> [LLM ERROR] {e}")
            llm_result = None
        if llm_result is not None:
            cache.set_llm(cache_key, llm_result)

    if not llm_result:
        result['llm_status'] = 'ERROR'
        return result

    geo_lat  = potential_events['ActionGeo_Lat'].iloc[0]  if not potential_events.empty else None
    geo_long = potential_events['ActionGeo_Long'].iloc[0] if not potential_events.empty else None
    status   = llm_result.get('status', 'ERROR')

    final_report = {
        "SQLDATE":   target_date,
        "City":      target_city,
        "FeatureID": int(feature_id),
        "Latitude":  float(geo_lat)  if geo_lat  is not None else None,
        "Longitude": float(geo_long) if geo_long is not None else None,
        "Summary":   llm_result.get('message', '메시지 없음'),
    }
    result['llm_status'] = status
    result['llm_report'] = json.dumps(final_report, ensure_ascii=False)
    result['is_anomaly'] = status in ['SUCCESS', 'AMBIGUOUS']
    return result


def verify_anomalies_with_llm(anomalies: pd.DataFrame, filtered_df: pd.DataFrame, url_df: pd.DataFrame, target_date: str, top_k: int = 20) -> pd.DataFrame:
    print(f"\n[TRACK 2] LLM Verification Started: {target_date} (top_k={top_k})")

    today_anomalies = anomalies[(anomalies['date'] == target_date) & (anomalies['is_anomaly'] == True)]
    top_targets = today_anomalies.sort_values('innov_z', ascending=False).head(top_k)

    anomalies['llm_status']   = 'UNVERIFIED'
    anomalies['llm_report']   = None
    anomalies['source_urls']  = None
    anomalies['scrape_stats'] = None

    if top_targets.empty:
        return anomalies

    rows = [(idx, row) for idx, row in top_targets.iterrows()]

    with ThreadPoolExecutor(max_workers=CITY_CONCURRENCY) as ex:
        fut_to_idx = {ex.submit(_process_city, row, filtered_df, url_df, target_date): idx for idx, row in rows}
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"[ERROR] city idx={idx} processing failed: {e}")
                continue

            # Flush buffered logs for this city (prevents interleaving across threads)
            for line in result['logs']:
                print(line)

            anomalies.at[idx, 'source_urls']  = result['source_urls']
            anomalies.at[idx, 'scrape_stats'] = result['scrape_stats']
            if result['llm_status'] is not None:
                anomalies.at[idx, 'llm_status'] = result['llm_status']
            if result['llm_report'] is not None:
                anomalies.at[idx, 'llm_report'] = result['llm_report']
            if not result['is_anomaly']:
                anomalies.at[idx, 'is_anomaly'] = False

    return anomalies