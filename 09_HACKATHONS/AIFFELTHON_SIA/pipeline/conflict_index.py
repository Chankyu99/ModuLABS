"""
SIA 갈등 지수 산출 및 이상 징후 탐지 엔진 (칼만 필터 기반)
──────────────────────────────────────────────────────────
- 갈등 지수(I) 산출 (빈도 x 파급력 x 심각도)
- 칼만 필터 혁신(Innovation) 추출 및 표준화 (Z-Score)
- 리스크 등급 분류 및 이상 징후 탐지
"""

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from pipeline.city_utils import canonicalize_city_by_feature_id, normalize_city_name
from pipeline.config import (
    LOGIT_WEIGHTS, ACTION_GEO_ALLOWED_COUNTRIES, CONFIRMED_CODES, MONITORED_COUNTRIES,
    KALMAN_Q_RATIO, KALMAN_R_RATIO, KALMAN_P0_RATIO, KALMAN_MIN_INIT_VAR,
    MIN_HISTORY, get_risk_level,
)

# ─── 1. 갈등 지수(I) 산출 로직 ──────────────────────────────

def compute_conflict_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    GDELT 원본 이벤트 데이터를 도시별 일별 갈등 지수(I)로 변환.

    per-event 점수는 로지스틱 회귀에서 학습한 계수(LOGIT_WEIGHTS)를 사용한다.
        score = const
              + w_log_sources  * Z(log1p(NumSources))
              + w_avg_tone     * Z(AvgTone)
              + w_avg_tone_sq  * Z(AvgTone²)
              + w_goldstein    * Z(GoldsteinScale)
    연속형 변수는 필터링 데이터 전체에서 mean/std를 구해 StandardScaler와 동일한
    방식으로 표준화한다(옵션 C). 도시·일자별 합계가 conflict_index가 된다.
    """
    if df.empty:
        return pd.DataFrame(columns=['date', 'city', 'country_code', 'lat', 'lon',
                                     'conflict_index', 'events', 'mentions', 'avg_tone'])

    # 모니터링 대상 국가, 분쟁 코드, 도시 단위 지오메트리(Type 4) 필터링
    # NumSources는 hard cutoff 대신 soft penalty로 반영한다.
    mask = (
        (df['Actor1CountryCode'].isin(MONITORED_COUNTRIES) | 
         df['Actor2CountryCode'].isin(MONITORED_COUNTRIES)) &
        pd.to_numeric(df['EventCode'], errors='coerce').isin(CONFIRMED_CODES) &
        (df['ActionGeo_Type'] == 4) &
        (df['ActionGeo_CountryCode'].isin(ACTION_GEO_ALLOWED_COUNTRIES)) &
        (df['NumSources'] >= 1)
    )
    filtered = df[mask].copy()

    if filtered.empty:
        return pd.DataFrame(columns=['date', 'city', 'country_code', 'lat', 'lon',
                                     'conflict_index', 'events', 'mentions', 'avg_tone'])

    # GDELT 중복 파싱 제거 로직 (Actor별 조합으로 뻥튀기된 동일 기사 제거)
    if 'Actor1Name' in filtered.columns and 'Actor2Name' in filtered.columns:
        filtered['info_count'] = filtered[['Actor1Name', 'Actor2Name']].notna().sum(axis=1)
        filtered = filtered.sort_values(by='info_count', ascending=False)
        filtered = filtered.drop_duplicates(
            subset=['SQLDATE', 'ActionGeo_FeatureID', 'EventCode', 'AvgTone', 'NumSources'], 
            keep='first'
        )

    # 날짜 형식 정리
    filtered['date'] = filtered['SQLDATE'].astype(str).str[:8]
    filtered['city'] = filtered['ActionGeo_FullName'].astype(str).map(normalize_city_name)
    filtered['city'] = canonicalize_city_by_feature_id(filtered['city'], filtered['ActionGeo_FeatureID'])

    # per-event feature 산출 (로지스틱 회귀 공식 입력)
    log_sources = np.log1p(pd.to_numeric(filtered['NumSources'], errors='coerce').fillna(0.0))
    avg_tone    = pd.to_numeric(filtered['AvgTone'], errors='coerce').fillna(0.0)
    avg_tone_sq = avg_tone ** 2
    goldstein   = pd.to_numeric(filtered['GoldsteinScale'], errors='coerce').fillna(0.0)

    # StandardScaler (옵션 C) — 필터링 데이터에서 즉석 mean/std 계산
    def _standardize(s: pd.Series) -> pd.Series:
        std = s.std(ddof=0)
        if std < 1e-10:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    z_log_sources = _standardize(log_sources)
    z_avg_tone    = _standardize(avg_tone)
    z_avg_tone_sq = _standardize(avg_tone_sq)
    z_goldstein   = _standardize(goldstein)

    # per-event 로지스틱 점수 → 시그모이드로 변환해 [0,1] 타격 확률로 사용
    # conflict_index는 이 확률의 합이므로 "기대 타격 건수"로 해석된다.
    w = LOGIT_WEIGHTS
    logit = (
        w['const']
        + w['log_sources']  * z_log_sources
        + w['avg_tone']     * z_avg_tone
        + w['avg_tone_sq']  * z_avg_tone_sq
        + w['goldstein']    * z_goldstein
    )
    filtered['weighted_mention'] = 1.0 / (1.0 + np.exp(-logit))

    # 도시별/일별로 데이터 집계
    agg = (
        filtered
        .groupby(['date', 'city'])
        .agg(
            country_code=('ActionGeo_CountryCode', 'first'),
            lat=('ActionGeo_Lat', 'median'),
            lon=('ActionGeo_Long', 'median'),
            conflict_index=('weighted_mention', 'sum'),
            events=('EventCode', 'count'),
            mentions=('NumMentions', 'sum'),
            avg_tone=('AvgTone', 'mean'),
        )
        .reset_index()
    )

    # 도시별 메타 정보 및 날짜 범위를 한 번에 집계
    city_meta = agg.groupby('city').agg(
        country_code=('country_code', 'first'),
        lat=('lat', 'median'),
        lon=('lon', 'median'),
        date_min=('date', 'min'),
        date_max=('date', 'max'),
    ).reset_index()

    # 도시별 날짜 범위만 grid로 확장 (전체 cartesian product 방지)
    grid = pd.concat([
        pd.DataFrame({
            'date': pd.date_range(r.date_min, r.date_max, freq='D').strftime('%Y%m%d').tolist(),
            'city': r.city,
        })
        for r in city_meta.itertuples()
    ], ignore_index=True)

    # merge로 빈 날짜 채우기 (loop 제거)
    result = grid.merge(agg, on=['city', 'date'], how='left')
    result = result.merge(
        city_meta[['city', 'country_code', 'lat', 'lon']],
        on='city', how='left', suffixes=('', '_meta'),
    )
    result['conflict_index'] = result['conflict_index'].fillna(0.0)
    result['events']         = result['events'].fillna(0).astype(int)
    result['mentions']       = result['mentions'].fillna(0).astype(int)
    result['avg_tone']       = result['avg_tone'].fillna(0.0)
    result['country_code']   = result['country_code'].fillna(result['country_code_meta'])
    result['lat']            = result['lat'].fillna(result['lat_meta'])
    result['lon']            = result['lon'].fillna(result['lon_meta'])
    result = result.drop(columns=['country_code_meta', 'lat_meta', 'lon_meta'])

    return result.sort_values(['date', 'city']).reset_index(drop=True)


# ─── 2. 칼만 필터 (예측 오차 추출) ──────────────────────────

def kalman_innovation(signal: np.ndarray,
                      Q: float = None,
                      R: float = None) -> dict:
    """
    1차원 칼만 필터를 적용하여 표준화된 예측 오차(Z-Score)를 추출.
    Z = (실측값 - 예측값) / sqrt(오차 공분산)
    """
    n = len(signal)
    if n < 2:
        return {'estimate': signal.copy(), 'innovation': np.zeros(n), 'norm_innov': np.zeros(n)}

    # 초기 30일 데이터를 기준으로 노이즈(Q, R) 자동 추정
    init_window = min(30, n)
    init_var = max(np.var(signal[:init_window]), KALMAN_MIN_INIT_VAR)

    # 파라미터가 없으면 설정값 비율에 따라 할당
    Q = Q if Q is not None else init_var * KALMAN_Q_RATIO
    R = R if R is not None else init_var * KALMAN_R_RATIO

    x_est = np.zeros(n)      # 상태 추정치
    P_est = np.zeros(n)      # 추정 불확실성 (공분산)
    innovation = np.zeros(n) # 예측 오차 (Residual)
    norm_innov = np.zeros(n) # 표준화된 예측 오차 (Z-Score)

    x_est[0] = signal[0]
    P_est[0] = R * KALMAN_P0_RATIO

    for t in range(1, n):
        # 1단계: 예측 (Predict)
        x_pred = x_est[t - 1]
        P_pred = P_est[t - 1] + Q

        # 2단계: 오차 확인 (Innovation)
        innovation[t] = signal[t] - x_pred
        S = P_pred + R  # 오차 공분산
        
        # 표준화 (Standardized Innovation)
        # norm_innov[t] = innovation[t] / max(np.sqrt(S), 1e-10)
        norm_innov[t] = np.clip(innovation[t] / max(np.sqrt(S), 1e-10), -10, 10)

        # 3단계: 업데이트 (Update)
        K = P_pred / S # 칼만 이득 (Kalman Gain)
        x_est[t] = x_pred + K * innovation[t]
        P_est[t] = (1 - K) * P_pred

    return {'estimate': x_est, 'innovation': innovation, 'norm_innov': norm_innov}


# ─── 3. 종합 이상 징후 탐지 ───────────────────────────────

def _process_city_kalman(args: tuple) -> pd.DataFrame | None:
    """도시 하나에 대해 칼만 필터 + 리스크 분류를 수행 (병렬 실행 단위)."""
    city, group = args
    group = group.sort_values('date').reset_index(drop=True)
    signal = group['conflict_index'].values.astype(float)

    if len(signal) < MIN_HISTORY:
        return None

    kf = kalman_innovation(signal)

    group = group.copy()
    group['kalman_est'] = kf['estimate']
    group['innovation'] = kf['innovation']
    group['innov_z']    = kf['norm_innov']

    risk_info          = group['innov_z'].apply(get_risk_level)
    group['risk_level'] = risk_info.apply(lambda x: x['level'])
    group['risk_label'] = risk_info.apply(lambda x: x['label'])
    group['risk_emoji'] = risk_info.apply(lambda x: x['emoji'])
    group['risk_guide'] = risk_info.apply(lambda x: x['guide'])
    group['is_anomaly'] = group['risk_level'] >= 1

    return group


def detect_anomalies(city_daily: pd.DataFrame,
                     target_date: str = None) -> pd.DataFrame:
    """
    집계된 갈등 지수 데이터를 처리하여 이상 징후와 리스크 레벨을 매칭.
    칼만 필터 Z-Score 단독으로 판정. 도시별 처리를 ThreadPoolExecutor로 병렬화.
    """
    city_groups = list(city_daily.groupby('city'))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = [r for r in executor.map(_process_city_kalman, city_groups) if r is not None]

    if not results:
        return pd.DataFrame()

    all_results = pd.concat(results, ignore_index=True)

    if target_date:
        all_results = all_results[all_results['date'] == target_date]

    return all_results.sort_values(['date', 'innov_z'], ascending=[True, False])
