"""
SIA 갈등 지수 산출 및 이상 징후 탐지 엔진 (칼만 필터 기반)
──────────────────────────────────────────────────────────
- 갈등 지수(I) 산출 (빈도 x 파급력 x 심각도)
- 칼만 필터 혁신(Innovation) 추출 및 표준화 (Z-Score)
- 리스크 등급 분류 및 이상 징후 탐지
"""

import numpy as np
import pandas as pd
from pipeline.config import (
    tone_weight, CONFIRMED_CODES, MONITORED_COUNTRIES,
    KALMAN_Q_RATIO, KALMAN_R_RATIO, KALMAN_P0_RATIO,
    MIN_HISTORY, get_risk_level,
)

# ─── 1. 갈등 지수(I) 산출 로직 ──────────────────────────────

def compute_conflict_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    GDELT 원본 이벤트 데이터를 도시별 일별 갈등 지수(I)로 변환.
    
    공식: I = Sigma(NumMentions * W(AvgTone))
    """
    if df.empty:
        return pd.DataFrame(columns=['date', 'city', 'conflict_index',
                                     'events', 'mentions', 'avg_tone'])

    # 모니터링 대상 국가, 분쟁 코드, 도시 단위 지오메트리(Type 4) 필터링
    mask = (
        (df['Actor1CountryCode'].isin(MONITORED_COUNTRIES) | 
         df['Actor2CountryCode'].isin(MONITORED_COUNTRIES)) &
        df['EventCode'].astype(str).str.split('.').str[0].isin(CONFIRMED_CODES) &
        (df['ActionGeo_Type'] == 4)
    )
    filtered = df[mask].copy()

    if filtered.empty:
        return pd.DataFrame(columns=['date', 'city', 'conflict_index',
                                     'events', 'mentions', 'avg_tone'])

    # 날짜 형식 정리 및 가중치 적용
    filtered['date'] = filtered['SQLDATE'].astype(str).str[:8]
    filtered['weight'] = filtered['AvgTone'].apply(tone_weight)
    filtered['weighted_mention'] = filtered['NumMentions'] * filtered['weight']

    # 도시별/일별로 데이터 집계
    agg = (
        filtered
        .groupby(['date', 'ActionGeo_FullName'])
        .agg(
            conflict_index=('weighted_mention', 'sum'),
            events=('EventCode', 'count'),
            mentions=('NumMentions', 'sum'),
            avg_tone=('AvgTone', 'mean'),
        )
        .reset_index()
        .rename(columns={'ActionGeo_FullName': 'city'})
    )

    return agg.sort_values(['date', 'city']).reset_index(drop=True)


# ─── 2. 칼만 필터 (예측 오차 추출) ──────────────────────────

def kalman_innovation(signal: np.ndarray,
                      Q: float = None,
                      R: float = None) -> dict:
    """
    1차원 칼만 필터를 적용하여 표준화된 예측 오차(Z-Score)를 추출.
    Z = (실측값 - 예측값) / sqrt(혁신 공분산)
    """
    n = len(signal)
    if n < 2:
        return {'estimate': signal.copy(), 'innovation': np.zeros(n), 'norm_innov': np.zeros(n)}

    # 초기 30일 데이터를 기준으로 노이즈(Q, R) 자동 추정
    init_window = min(30, n)
    init_var = max(np.var(signal[:init_window]), 1e-6)

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
        norm_innov[t] = innovation[t] / max(np.sqrt(S), 1e-10)

        # 3단계: 업데이트 (Update)
        K = P_pred / S # 칼만 이득 (Kalman Gain)
        x_est[t] = x_pred + K * innovation[t]
        P_est[t] = (1 - K) * P_pred

    return {'estimate': x_est, 'innovation': innovation, 'norm_innov': norm_innov}


# ─── 3. 종합 이상 징후 탐지 ───────────────────────────────

def detect_anomalies(city_daily: pd.DataFrame,
                     target_date: str = None) -> pd.DataFrame:
    """
    집계된 갈등 지수 데이터를 처리하여 이상 징후와 리스크 레벨을 매칭.
    """
    results = []

    for city, group in city_daily.groupby('city'):
        group = group.sort_values('date').reset_index(drop=True)
        signal = group['conflict_index'].values.astype(float)

        # 충분한 통계 데이터(30일)가 쌓이지 않은 경우 스킵
        if len(signal) < MIN_HISTORY:
            continue

        # 칼만 필터 실행
        kf = kalman_innovation(signal)
        
        group = group.copy()
        group['kalman_est'] = kf['estimate']
        group['innovation'] = kf['innovation']
        group['innov_z'] = kf['norm_innov']

        # 리스크 등급 분류
        risk_info = group['innov_z'].apply(get_risk_level)
        group['risk_level']  = risk_info.apply(lambda x: x['level'])
        group['risk_label']  = risk_info.apply(lambda x: x['label'])
        group['risk_emoji']  = risk_info.apply(lambda x: x['emoji'])
        group['risk_guide']  = risk_info.apply(lambda x: x['guide'])
        group['is_anomaly']  = group['risk_level'] >= 1

        results.append(group)

    if not results:
        return pd.DataFrame()

    all_results = pd.concat(results, ignore_index=True)

    # 특정 날짜가 지정된 경우 필터링
    if target_date:
        all_results = all_results[all_results['date'] == target_date]

    return all_results.sort_values(['date', 'innov_z'], ascending=[True, False])
