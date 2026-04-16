"""
SIA GDELT 1.0 이벤트 데이터 수량 모듈
───────────────────────────────────
GDELT 1.0 마스터 리스트에서 일별 이벤트 CSV를 다운로드하고,
관심 지역(ROI) 필터링을 적용하여 로컬에 저장합니다.
"""

import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from pipeline.config import (
    ACTION_GEO_ALLOWED_COUNTRIES, DATA_DIR, PARQUET_PATH, CONFIRMED_CODES, MONITORED_COUNTRIES,
)

# GDELT 1.0 마스터 파일 리스트 URL
GDELT_MASTER_URL = "http://data.gdeltproject.org/events/index.html"
GDELT_BASE_URL = "http://data.gdeltproject.org/events/"
HISTORY_START_DATE = 20260000

# GDELT 1.0 이벤트 주요 컬럼 (58개)
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
    'Actor1Geo_ADM1Code', 'Actor1Geo_Lat', 'Actor1Geo_Long', 'Actor1Geo_FeatureID',
    'Actor2Geo_Type', 'Actor2Geo_FullName', 'Actor2Geo_CountryCode',
    'Actor2Geo_ADM1Code', 'Actor2Geo_Lat', 'Actor2Geo_Long', 'Actor2Geo_FeatureID',
    'ActionGeo_Type', 'ActionGeo_FullName', 'ActionGeo_CountryCode',
    'ActionGeo_ADM1Code', 'ActionGeo_Lat', 'ActionGeo_Long', 'ActionGeo_FeatureID',
    'DATEADDED', 'SOURCEURL',
]

def apply_monitored_filter(df: pd.DataFrame) -> pd.DataFrame:
    """모니터링 국가, 분쟁 코드, 도시 단위(ActionGeo_Type=4) 필터 적용"""
    mask = (
        (df['Actor1CountryCode'].isin(MONITORED_COUNTRIES) |
         df['Actor2CountryCode'].isin(MONITORED_COUNTRIES)) &
        pd.to_numeric(df['EventCode'], errors='coerce').isin(CONFIRMED_CODES) &
        (df['ActionGeo_Type'] == 4) &
        (df['ActionGeo_CountryCode'].isin(ACTION_GEO_ALLOWED_COUNTRIES))
    )
    return df[mask].copy()

def fetch_daily(target_date: str, save: bool = True) -> pd.DataFrame:
    """특정 날짜의 GDELT v1.0 이벤트를 다운로드하고 필터링하여 수집"""
    print(f"--- GDELT 데이터 수집 시작: {target_date} ---")

    save_path = DATA_DIR / f"{target_date}.parquet"
    if save_path.exists():
        print(f"  [로컬 경로] {save_path.name} 파일이 이미 존재합니다.")
        return pd.read_parquet(save_path)

    # GDELT v1.0: 일별 단일 파일 (YYYYMMDD.export.CSV.zip)
    url = f"{GDELT_BASE_URL}{target_date}.export.CSV.zip"
    print(f"  [다운로드] {url}")

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except requests.exceptions.HTTPError:
        print(f"  [오류] {target_date}에 해당하는 GDELT 파일이 없습니다.")
        return pd.DataFrame()

    dfs = []
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                batch = pd.read_csv(f, sep='\t', header=None, names=GDELT_COLUMNS, 
                                    dtype=str, on_bad_lines='skip')
                
                # 수치형 데이터 변환
                for c in ['NumMentions', 'NumSources', 'NumArticles', 'ActionGeo_Type']:
                    batch[c] = pd.to_numeric(batch[c], errors='coerce').fillna(0).astype(int)
                for c in ['AvgTone', 'GoldsteinScale', 'ActionGeo_Lat', 'ActionGeo_Long']:
                    batch[c] = pd.to_numeric(batch[c], errors='coerce').fillna(0.0)
                
                dfs.append(batch)
    except Exception as e:
        print(f"  [오류] 파일 처리 실패: {e}")
        return pd.DataFrame()

    if not dfs: return pd.DataFrame()

    # 데이터 병합 및 필터링
    all_data = pd.concat(dfs, ignore_index=True)
    filtered = apply_monitored_filter(all_data)
    # 히스토리 데이터와의 호환성을 위해 콤마 이후의 세부 지명 제거
    filtered['ActionGeo_FullName'] = filtered['ActionGeo_FullName'].astype(str).apply(lambda x: x.split(',')[0])
    print(f"  [완료] 총 {len(filtered):,}건의 유효한 이벤트를 선별했습니다.")

    # 저장
    if save and not filtered.empty:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        filtered.to_parquet(save_path, index=False)

    return filtered


def _history_cache_path() -> Path:
    """2026년 전용 히스토리 캐시 경로."""
    return PARQUET_PATH.with_name("gdelt_main_2026.parquet")


def _load_history_2026() -> pd.DataFrame:
    """2026년 히스토리 전용 캐시를 로드하거나 최초 1회 생성한다."""
    cache_path = _history_cache_path()
    if cache_path.exists():
        print(f"  [히스토리 요약] {cache_path.name} 로드 중...")
        return pd.read_parquet(cache_path)

    if not PARQUET_PATH.exists():
        return pd.DataFrame()

    print(f"  [히스토리 요약] {PARQUET_PATH.name}에서 2026년 전용 캐시 생성 중...")
    try:
        history = pd.read_parquet(
            PARQUET_PATH,
            filters=[("SQLDATE", ">=", HISTORY_START_DATE)],
        )
    except Exception:
        history = pd.read_parquet(PARQUET_PATH)
        if "SQLDATE" in history.columns:
            sql_dates = pd.to_numeric(history["SQLDATE"], errors="coerce")
            history = history[sql_dates >= HISTORY_START_DATE].copy()

    history.to_parquet(cache_path, index=False)
    print(f"  [히스토리 요약] {cache_path.name} 캐시 저장 완료 ({len(history):,}건)")
    return history

def load_all_data(include_daily: bool = True, target_date: Optional[str] = None) -> pd.DataFrame:
    """기존 대용량 Parquet 파일과 로컬 일별 파일들을 병합하여 로드한다.

    Args:
        include_daily: data/daily parquet 포함 여부
        target_date: YYYYMMDD. 지정 시 해당 날짜 이하의 일별 parquet만 로드
    """
    dfs = []
    
    # 1. 히스토리 데이터 로드
    if PARQUET_PATH.exists():
        history_2026 = _load_history_2026()
        if not history_2026.empty:
            dfs.append(history_2026)

    # 2. 일별 수집 데이터 로드
    if include_daily and DATA_DIR.exists():
        all_daily_files = sorted(DATA_DIR.glob("*.parquet"))
        daily_files = []
        skipped_future_files = []

        for path in all_daily_files:
            try:
                file_date = path.stem[:8]
            except Exception:
                continue

            if not (len(file_date) == 8 and file_date.isdigit()):
                continue

            if target_date and file_date > target_date:
                skipped_future_files.append(path.name)
                continue

            daily_files.append(path)

        if daily_files:
            print(f"  [일별 데이터] {len(daily_files)}개 파일을 병합 중...")
            if skipped_future_files:
                print(f"  [일별 데이터] 미래 기준 파일 {len(skipped_future_files)}개 제외")
            for f in daily_files:
                dfs.append(pd.read_parquet(f))

    if not dfs: return pd.DataFrame()

    # 3. 병합 및 중복 제거
    combined = pd.concat(dfs, ignore_index=True)
    if 'GLOBALEVENTID' in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(subset=['GLOBALEVENTID'])
        after = len(combined)
        if before > after:
            print(f"  [중복 제거] {before:,} -> {after:,} 건으로 압축 완료.")

    # 4. 연산 최적화: 2026년 데이터만 필터링 (과거 13년치 제거)
    if 'SQLDATE' in combined.columns:
        combined = combined[pd.to_numeric(combined['SQLDATE'], errors='coerce') >= HISTORY_START_DATE].copy()
        if target_date:
            sql_dates = pd.to_numeric(combined['SQLDATE'], errors='coerce')
            combined = combined[sql_dates <= int(target_date)].copy()
        
    return combined
