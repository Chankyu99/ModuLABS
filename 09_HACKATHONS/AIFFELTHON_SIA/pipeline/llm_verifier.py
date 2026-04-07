"""
SIA 갈등 모니터링 파이프라인 - LLM 게이트키퍼 (Hybrid v2)
─────────────────────────────────────────────────────────
하이브리드 검증 파이프라인:
  1. 블랙리스트 사전 제거 (비용 0)
  2. OG 메타 + JSON-LD 구조화 데이터 파싱 (비용 0)
  3. Local NER: 제목/리드에서 GPE(지명) 후보 추출 (비용 0)
  4. LLM 최종 검증: GPE 리스트 + 제목만 전달 → 토큰 최소화
"""

import json
import os
import re
import time
from typing import Optional

import pandas as pd
import requests

from pipeline.config import (
    CITY_BLACKLIST, LLM_MODELS, LLM_TOP_N,
    LLM_TOP_K_URLS, LLM_CONFIDENCE_THRESHOLD,
    CONFIRMED_CODES, MONITORED_COUNTRIES,
)


# ─── 1. 블랙리스트 필터링 (비용 0) ──────────────────────

def filter_blacklist(anomalies: pd.DataFrame) -> pd.DataFrame:
    """블랙리스트에 등재된 도시(조직명/무기명) 제거"""
    mask = anomalies['city'].apply(
        lambda c: not any(bl in c.split(',')[0] for bl in CITY_BLACKLIST)
    )
    removed = anomalies[~mask]
    if not removed.empty:
        names = ', '.join(removed['city'].apply(lambda c: c.split(',')[0]).unique())
        print(f"  [블랙리스트] 제거: {names}")
    return anomalies[mask].copy()


# ─── 2. SOURCEURL 추출 ──────────────────────────────────

def extract_urls_for_cities(
    anomalies: pd.DataFrame,
    raw: pd.DataFrame,
    url_df: pd.DataFrame,
    target_date: str,
    top_n: int = LLM_TOP_N,
    top_k: int = LLM_TOP_K_URLS,
) -> dict:
    """이상 징후 상위 N개 도시에 대해 각 도시의 상위 K개 SOURCEURL 추출."""
    today = anomalies[anomalies['date'] == target_date]
    alerts = today[today['is_anomaly'] == True].sort_values('innov_z', ascending=False)
    top_cities = alerts.head(top_n)

    mask = (
        (raw['Actor1CountryCode'].isin(MONITORED_COUNTRIES) |
         raw['Actor2CountryCode'].isin(MONITORED_COUNTRIES)) &
        raw['EventCode'].astype(str).str.split('.').str[0].isin(CONFIRMED_CODES) &
        (raw['ActionGeo_Type'] == 4)
    )
    filtered = raw[mask].copy()
    filtered['date'] = filtered['SQLDATE'].astype(str).str[:8]
    day_events = filtered[filtered['date'] == target_date]

    result = {}
    for _, row in top_cities.iterrows():
        city = row['city']
        city_events = day_events[day_events['ActionGeo_FullName'] == city]
        if city_events.empty:
            result[city] = []
            continue

        top_events = city_events.nlargest(top_k, 'NumSources')
        event_ids = top_events['GLOBALEVENTID'].values
        urls = url_df[url_df['GLOBALEVENTID'].isin(event_ids)].drop_duplicates('GLOBALEVENTID')
        merged = top_events[['GLOBALEVENTID', 'NumMentions', 'NumSources']].merge(
            urls[['GLOBALEVENTID', 'SOURCEURL']], on='GLOBALEVENTID', how='left'
        )

        city_urls = []
        for _, u in merged.iterrows():
            url = u.get('SOURCEURL', '')
            if pd.isna(url) or not url:
                continue
            city_urls.append({
                'url': url,
                'sources': int(u['NumSources']),
                'mentions': int(u['NumMentions']),
            })
        result[city] = city_urls

    return result


# ─── 3. 기사 메타데이터 파싱 (OG + JSON-LD) ──────────────

def _parse_og_meta(html: str) -> dict:
    """OpenGraph 메타 태그 파싱"""
    result = {}
    for prop in ['title', 'description']:
        # property="og:..." content="..."
        m = re.search(
            rf'<meta\s+(?:property|name)=["\']og:{prop}["\']\s+content=["\']([^"\']+)["\']',
            html, re.IGNORECASE
        )
        if not m:
            m = re.search(
                rf'<meta\s+content=["\']([^"\']+)["\']\s+(?:property|name)=["\']og:{prop}["\']',
                html, re.IGNORECASE
            )
        if m:
            result[prop] = m.group(1).strip()
    return result


def _parse_json_ld(html: str) -> list:
    """JSON-LD 구조화 데이터에서 장소 정보 추출"""
    places = []
    for m in re.finditer(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
                         html, re.DOTALL | re.IGNORECASE):
        try:
            data = json.loads(m.group(1))
            # 단일 객체 또는 리스트
            items = data if isinstance(data, list) else [data]
            for item in items:
                # contentLocation, locationCreated
                for loc_key in ['contentLocation', 'locationCreated', 'location']:
                    loc = item.get(loc_key, {})
                    if isinstance(loc, dict):
                        name = loc.get('name', '')
                        if name:
                            places.append(name)
                    elif isinstance(loc, str):
                        places.append(loc)
                # about 필드
                about = item.get('about', [])
                if isinstance(about, list):
                    for a in about:
                        if isinstance(a, dict) and a.get('@type') in ('Place', 'City', 'Country'):
                            places.append(a.get('name', ''))
        except (json.JSONDecodeError, AttributeError):
            pass
    return [p for p in places if p]


def fetch_article_info(url: str, timeout: int = 5) -> dict:
    """
    URL에서 기사 정보를 구조적으로 추출.
    OG 메타 → JSON-LD → <title> → meta description 순으로 파싱.
    """
    info = {'title': '', 'lead': '', 'source': url, 'json_ld_places': []}

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
            'Accept': 'text/html',
        }
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        html = resp.text[:30000]  # 상위 30KB

        # OG 메타
        og = _parse_og_meta(html)
        info['title'] = og.get('title', '')
        info['lead'] = og.get('description', '')

        # <title> 폴백
        if not info['title']:
            m = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
            if m:
                info['title'] = m.group(1).strip()

        # meta description 폴백
        if not info['lead']:
            m = re.search(
                r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']+)["\']',
                html, re.IGNORECASE
            )
            if m:
                info['lead'] = m.group(1).strip()

        # JSON-LD 장소 정보 (비용 0 지명 추출)
        info['json_ld_places'] = _parse_json_ld(html)

    except Exception:
        pass

    return info


# ─── 4. Local NER: 지명(GPE) 후보 추출 (비용 0) ─────────

# 중동·이란 주요 도시 사전
_KNOWN_CITIES = {
    'Tehran', 'Isfahan', 'Tabriz', 'Shiraz', 'Mashhad', 'Kermanshah',
    'Natanz', 'Fordow', 'Arak', 'Bushehr', 'Bandar Abbas', 'Kharg',
    'Minab', 'Qom', 'Ahvaz', 'Zanjan', 'Semnan', 'Parchin',
    'Beirut', 'Baghdad', 'Damascus', 'Riyadh', 'Doha', 'Dubai',
    'Jerusalem', 'Tel Aviv', 'Gaza', 'Haifa', 'Cairo', 'Amman',
    'Hormuz', 'Kharg Island', 'Strait of Hormuz', 'Gulf of Oman',
    'Homs', 'Aleppo', 'Mosul', 'Basra', 'Karbala', 'Erbil',
    'Ramallah', 'Nablus', 'Hebron', 'Jenin', 'Rafah',
    'Iran', 'Iraq', 'Israel', 'Lebanon', 'Syria', 'Saudi Arabia',
    'Yemen', 'Bahrain', 'Qatar', 'UAE', 'Jordan', 'Egypt', 'Turkey',
}

# 대문자로 시작하는 1~3단어 패턴 (간이 NER)
_GPE_PATTERN = re.compile(
    r'\b([A-Z][a-z]+(?:\s+(?:of|al-|el-|Al-|El-)?[A-Z][a-z]+){0,2})\b'
)


def extract_local_gpe(text: str) -> list:
    """
    Local NER: 텍스트에서 지명 후보를 추출 (비용 0).
    1. 기존 도시 사전 매칭
    2. 대문자 시작 패턴(간이 NER)
    3. JSON-LD에서 추출된 장소
    """
    found = set()

    # 사전 매칭
    for city in _KNOWN_CITIES:
        if city.lower() in text.lower():
            found.add(city)

    # 패턴 매칭 (대문자 시작 단어 조합)
    for m in _GPE_PATTERN.finditer(text):
        candidate = m.group(1)
        # 일반 단어 필터링
        if candidate not in ('The', 'This', 'That', 'With', 'From', 'After',
                             'Before', 'About', 'Trump', 'Biden', 'Netanyahu',
                             'Khamenei', 'Obama', 'Putin', 'Reuters', 'Associated',
                             'Press', 'Monday', 'Tuesday', 'Wednesday', 'Thursday',
                             'Friday', 'Saturday', 'Sunday', 'January', 'February',
                             'March', 'April', 'May', 'June', 'July', 'August',
                             'September', 'October', 'November', 'December',
                             'New', 'United', 'States', 'Middle', 'East',
                             'Breaking', 'Live', 'Update', 'Latest', 'Watch',
                             'Video', 'Photo', 'Images', 'Read', 'More',
                             'World', 'News', 'War', 'Attack', 'Strike'):
            found.add(candidate)

    return sorted(found)


# ─── 5. LLM 배치 검증 (토큰 최소화) ─────────────────────

def build_verification_prompt(city_data: dict) -> str:
    """
    도시별 기사 정보 + Local NER 결과를 최소 토큰으로 구성.
    LLM에게는 GPE 리스트 + 제목만 전달하여 검증 요청.
    """
    lines = [
        "You are a GDELT geocoding accuracy analyst.\n"
        "For each city below, I provide news article titles, leads, and "
        "locally extracted place names (GPE). Determine if the articles "
        "actually report events IN or ABOUT that specific city.\n\n"
        "RULES (apply in order):\n"
        "1. CROSS-CHECK: If the GPE list shows a DIFFERENT country as the "
        "event location (e.g. city='Khondab, Iran' but GPE shows 'Saudi Arabia' "
        "as the attack target) → IRRELEVANT, even if the city itself is strategic.\n"
        "2. DIRECT MENTION: If the city name appears in the article title or lead → RELEVANT.\n"
        "3. STRATEGIC INFERENCE: If articles discuss country-wide military operations "
        "(e.g. 'US strikes Iran') AND the city has known military/nuclear/energy "
        "infrastructure → RELEVANT. The city need not be explicitly named.\n"
        "4. ORGANIZATION/WEAPON: If the 'city' is actually an organization name "
        "(e.g. Hezbollah), weapon name (e.g. Shahed drone), or region → IRRELEVANT.\n"
        "5. INSUFFICIENT: Cannot determine from given data.\n"
    ]

    for city, articles in city_data.items():
        city_short = city.split(',')[0]
        lines.append(f"\n■ City: {city_short}")

        for i, art in enumerate(articles, 1):
            title = art.get('title', '(no title)')[:120]
            lead = art.get('lead', '')[:200]
            gpe = art.get('gpe', [])
            ld_places = art.get('json_ld_places', [])

            lines.append(f"  Art{i}: \"{title}\"")
            if lead:
                lines.append(f"    Lead: \"{lead}\"")
            if gpe:
                lines.append(f"    GPE: {', '.join(gpe[:10])}")
            if ld_places:
                lines.append(f"    JSON-LD: {', '.join(ld_places[:5])}")

    lines.append(
        "\n\nRespond ONLY with a JSON array. "
        "'confidence' = probability that a real event ACTUALLY OCCURRED in this city "
        "(0.0 = definitely NOT in this city, 1.0 = definitely in this city). "
        "If CROSS-CHECK shows events in a different country, confidence MUST be ≤ 0.1.\n"
        "[{\"city\": \"name\", \"relevant\": N, \"total\": N, "
        "\"confidence\": 0.0~1.0, \"reason\": \"one line\"}]"
    )
    return '\n'.join(lines)


def call_gemini(prompt: str) -> Optional[list]:
    """Gemini API 호출. 모델 우선순위에 따라 폴백."""
    from google import genai
    from google.genai import types

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith('GEMINI_API_KEY'):
                        api_key = line.split('=', 1)[1].strip().strip('"\'')
                        break

    if not api_key:
        print("  [LLM] ⚠️ GEMINI_API_KEY가 설정되지 않았습니다.")
        return None

    client = genai.Client(api_key=api_key)

    for model in LLM_MODELS:
        try:
            print(f"  [LLM] 모델: {model}")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                )
            )

            text = response.text.strip()
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(text)

        except Exception as e:
            print(f"  [LLM] ⚠️ {model} 실패: {e}")
            if model != LLM_MODELS[-1]:
                print(f"  [LLM] 다음 모델로 폴백...")
            time.sleep(1)

    print("  [LLM] ❌ 모든 모델 실패. 검증 생략.")
    return None


# ─── 6. 통합 검증 함수 ───────────────────────────────────

def verify_top_cities(
    anomalies: pd.DataFrame,
    raw: pd.DataFrame,
    url_df: pd.DataFrame,
    target_date: str,
) -> pd.DataFrame:
    """
    하이브리드 LLM 게이트키퍼 파이프라인:
      1. 블랙리스트 제거 (비용 0)
      2. SOURCEURL 추출
      3. OG + JSON-LD 파싱 (비용 0)
      4. Local NER: 지명 후보 추출 (비용 0)
      5. LLM 최종 검증: GPE + 제목만 전달 (최소 토큰)
      6. 신뢰도 점수 반영
    """
    print(f"\n  ── LLM 게이트키퍼 가동 (Hybrid v2) ──")

    # 1. 블랙리스트
    anomalies = filter_blacklist(anomalies)

    anomalies['llm_confidence'] = -1.0
    anomalies['llm_reason'] = ''

    # 2. SOURCEURL 추출
    today_alerts = anomalies[
        (anomalies['date'] == target_date) & (anomalies['is_anomaly'] == True)
    ].sort_values('innov_z', ascending=False).head(LLM_TOP_N)

    if today_alerts.empty:
        print("  [LLM] 검증 대상 도시 없음")
        return anomalies

    city_urls = extract_urls_for_cities(anomalies, raw, url_df, target_date)

    # 3~4. 기사 파싱 + Local NER
    print(f"  [LLM] {len(city_urls)}개 도시 기사 파싱 및 NER...")
    city_data = {}

    for city, urls in city_urls.items():
        if not urls:
            continue

        articles = []
        for u in urls:
            info = fetch_article_info(u['url'])
            info['sources'] = u['sources']
            info['mentions'] = u['mentions']

            # Local NER: 제목 + 리드에서 지명 추출
            text = f"{info['title']} {info['lead']}"
            info['gpe'] = extract_local_gpe(text)

            if not info['title'] and not info['lead']:
                continue
            articles.append(info)

        city_short = city.split(',')[0]
        gpe_all = set()
        for a in articles:
            gpe_all.update(a.get('gpe', []))

        ld_all = set()
        for a in articles:
            ld_all.update(a.get('json_ld_places', []))

        gpe_str = ', '.join(sorted(gpe_all)[:8]) if gpe_all else '(없음)'
        print(f"    {city_short:20s} | 기사 {len(articles)}건 | GPE: {gpe_str}")

        if articles:
            city_data[city] = articles

    if not city_data:
        print("  [LLM] 파싱된 기사 없음. 검증 생략.")
        return anomalies

    # 5. LLM 배치 검증
    prompt = build_verification_prompt(city_data)
    prompt_tokens = len(prompt.split())
    print(f"  [LLM] Gemini 검증 요청 ({len(city_data)}개 도시, ~{prompt_tokens} words)...")
    llm_result = call_gemini(prompt)

    if llm_result is None:
        print("  [LLM] 검증 실패. 기존 결과 유지.")
        return anomalies

    # 6. 결과 반영
    print(f"\n  ── LLM 검증 결과 ──")
    for item in llm_result:
        city_name = item.get('city', '')
        confidence = item.get('confidence', -1)
        reason = item.get('reason', '')

        mask = (
            (anomalies['date'] == target_date) &
            anomalies['city'].str.contains(city_name, case=False, na=False)
        )
        anomalies.loc[mask, 'llm_confidence'] = confidence
        anomalies.loc[mask, 'llm_reason'] = reason

        if confidence >= 0:
            emoji = '✅' if confidence >= LLM_CONFIDENCE_THRESHOLD else '⚠️'
            level = '높음' if confidence >= 0.7 else '보통' if confidence >= LLM_CONFIDENCE_THRESHOLD else '낮음'
            print(f"    {emoji} {city_name:20s} | 신뢰도 {confidence:.0%} ({level}) | {reason}")

    return anomalies
