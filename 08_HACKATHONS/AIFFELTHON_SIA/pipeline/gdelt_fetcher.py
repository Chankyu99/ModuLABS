"""
GDELT 2.0 Event 데이터 수집 모듈
──────────────────────────────────
GDELT 2.0 마스터 파일 리스트에서 일별 Event CSV를 다운로드하고,
Triad 필터를 적용하여 로컬에 저장.
"""

import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from pipeline.config import (
    DATA_DIR, PARQUET_PATH, CONFIRMED_CODES, TRIAD_COUNTRIES,
)

# GDELT 2.0 Event 파일 URL 패턴
GDELT_MASTER_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
GDELT_EVENT_BASE = "http://data.gdeltproject.org/gdeltv2/"

# GDELT 2.0 Event 컬럼 (61개 중 사용하는 것만 정의)
GDELT_COLUMNS = [
    'GLOBALEVENTID', 'SQLDATE', 'MonthYear', 'Year', 'FractionDate',
    'Actor1Code', 'Actor1Name', 'Actor1CountryCode', 'Actor1KnownGroupCode',
    'Actor1EthnicCode', 'Actor1Religion1Code', 'Actor1Religion2Code',
    'Actor1Type1Code', 'Actor1Type2Code', 'Actor1Type3Code',
    'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2KnownGroupCode',
    'Actor2EthnicCode', 'Actor2Religion1Code', 'Actor2Religion2Code',
    'Actor2Type1Code', 'Actor2Type2Code', 'Actor2Type3Code',
    'IsRootEvent', 'EventCode', 'EventBaseCode', 'EventRootCode',
    'QuadClass', 'GoldsteinScale', 'NumMentions', 'NumSources',
    'NumArticles', 'AvgTone',
    'Actor1Geo_Type', 'Actor1Geo_FullName', 'Actor1Geo_CountryCode',
    'Actor1Geo_ADM1Code', 'Actor1Geo_ADM2Code',
    'Actor1Geo_Lat', 'Actor1Geo_Long', 'Actor1Geo_FeatureID',
    'Actor2Geo_Type', 'Actor2Geo_FullName', 'Actor2Geo_CountryCode',
    'Actor2Geo_ADM1Code', 'Actor2Geo_ADM2Code',
    'Actor2Geo_Lat', 'Actor2Geo_Long', 'Actor2Geo_FeatureID',
    'ActionGeo_Type', 'ActionGeo_FullName', 'ActionGeo_CountryCode',
    'ActionGeo_ADM1Code', 'ActionGeo_ADM2Code',
    'ActionGeo_Lat', 'ActionGeo_Long', 'ActionGeo_FeatureID',
    'DATEADDED', 'SOURCEURL',
]


def get_event_urls_for_date(target_date: str) -> list:
    """
    특정 날짜의 GDELT 2.0 Event 파일 URL 목록을 반환.

    Parameters
    ----------
    target_date : str
        'YYYYMMDD' 형식

    Returns
    -------
    list of str : URL 목록
    """
    print(f"  📡 GDELT 마스터 파일 조회 중...")
    resp = requests.get(GDELT_MASTER_URL, timeout=30)
    resp.raise_for_status()

    urls = []
    for line in resp.text.strip().split('\n'):
        parts = line.strip().split()
        if len(parts) >= 3:
            url = parts[2]
            if '.export.CSV.zip' in url and target_date in url:
                urls.append(url)

    return urls


def download_and_parse(url: str) -> pd.DataFrame:
    """
    GDELT Event CSV.zip 파일을 다운로드하고 파싱.

    Returns
    -------
    pd.DataFrame : 원본 이벤트 데이터
    """
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(
                f, sep='\t', header=None,
                names=GDELT_COLUMNS,
                dtype=str,
                on_bad_lines='skip',
            )

    # 숫자형 변환
    for col in ['NumMentions', 'NumSources', 'NumArticles',
                 'ActionGeo_Type', 'QuadClass']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    for col in ['AvgTone', 'GoldsteinScale', 'ActionGeo_Lat', 'ActionGeo_Long']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    return df


def apply_triad_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Triad 국가 + 확정 코드 필터 적용"""
    mask = (
        df['Actor1CountryCode'].isin(TRIAD_COUNTRIES) &
        df['Actor2CountryCode'].isin(TRIAD_COUNTRIES) &
        df['EventCode'].isin(CONFIRMED_CODES) &
        (df['ActionGeo_Type'] == 4)
    )
    return df[mask].copy()


def fetch_daily(target_date: str, save: bool = True) -> pd.DataFrame:
    """
    특정 날짜의 GDELT 데이터를 수집하고 Triad 필터 적용.

    Parameters
    ----------
    target_date : str
        'YYYYMMDD' 형식
    save : bool
        True면 data/daily/ 에 parquet로 저장

    Returns
    -------
    pd.DataFrame : 필터링된 이벤트 데이터
    """
    print(f"\n{'─'*60}")
    print(f"  📥 {target_date} GDELT 데이터 수집")
    print(f"{'─'*60}")

    # 이미 있는지 확인
    save_path = DATA_DIR / f"{target_date}.parquet"
    if save_path.exists():
        print(f"  ✅ 이미 존재: {save_path.name}")
        return pd.read_parquet(save_path)

    # URL 수집
    urls = get_event_urls_for_date(target_date)
    if not urls:
        print(f"  ⚠️  {target_date}에 해당하는 GDELT 파일 없음")
        return pd.DataFrame()

    print(f"  📦 {len(urls)}개 파일 다운로드 중...")

    # 다운로드 & 파싱
    dfs = []
    for i, url in enumerate(urls):
        try:
            df = download_and_parse(url)
            dfs.append(df)
            if (i + 1) % 10 == 0:
                print(f"     {i+1}/{len(urls)} 완료...")
        except Exception as e:
            print(f"     ⚠️ {url.split('/')[-1]} 실패: {e}")

    if not dfs:
        return pd.DataFrame()

    all_data = pd.concat(dfs, ignore_index=True)
    print(f"  📊 전체: {len(all_data):,}건")

    # Triad 필터
    filtered = apply_triad_filter(all_data)
    print(f"  🔍 Triad 필터 후: {len(filtered):,}건")

    # 저장
    if save and not filtered.empty:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        filtered.to_parquet(save_path, index=False)
        print(f"  💾 저장: {save_path.name}")

    return filtered


def load_historical(parquet_path: str = None) -> pd.DataFrame:
    """
    기존 parquet 파일 로드 (2025~2026 Q1 데이터)

    Returns
    -------
    pd.DataFrame
    """
    path = Path(parquet_path) if parquet_path else PARQUET_PATH
    if not path.exists():
        raise FileNotFoundError(f"parquet 파일 없음: {path}")

    print(f"  📂 기존 데이터 로드: {path.name}")
    df = pd.read_parquet(path)
    print(f"     {len(df):,}건 로드 완료")
    return df


def load_all_data(include_daily: bool = True) -> pd.DataFrame:
    """
    기존 parquet + data/daily/*.parquet를 합쳐서 반환.

    Parameters
    ----------
    include_daily : bool
        True면 data/daily/ 의 일별 파일도 합침

    Returns
    -------
    pd.DataFrame
    """
    dfs = []

    # 기존 데이터
    if PARQUET_PATH.exists():
        dfs.append(load_historical())

    # 일별 수집 데이터
    if include_daily and DATA_DIR.exists():
        daily_files = sorted(DATA_DIR.glob("*.parquet"))
        if daily_files:
            print(f"  📂 일별 데이터 {len(daily_files)}개 로드 중...")
            for f in daily_files:
                dfs.append(pd.read_parquet(f))

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # 중복 제거 (GLOBALEVENTID 기준)
    if 'GLOBALEVENTID' in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(subset=['GLOBALEVENTID'])
        after = len(combined)
        if before > after:
            print(f"  🔄 중복 제거: {before:,} → {after:,}")

    return combined
