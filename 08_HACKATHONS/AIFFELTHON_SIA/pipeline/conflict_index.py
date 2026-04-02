"""
갈등지수(I) 산출 + 칼만 필터 Innovation + Rolling Z-score
──────────────────────────────────────────────────────────
핵심 로직 모듈. 외부 의존성 없이 numpy + pandas만 사용.
"""

import numpy as np
import pandas as pd
from pipeline.config import (
    tone_weight, CONFIRMED_CODES, TRIAD_COUNTRIES,
    KALMAN_Q_RATIO, KALMAN_R_RATIO, KALMAN_P0_RATIO,
    ROLLING_WINDOW, Z_THRESHOLD, MIN_HISTORY,
)


# ──────────────────────────────────────────────
# 1. 갈등지수 I 산출
# ──────────────────────────────────────────────
def compute_conflict_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    이벤트 단위 DataFrame → 도시별 일별 갈등지수 I

    Parameters
    ----------
    df : pd.DataFrame
        최소 컬럼: SQLDATE, EventCode, Actor1CountryCode, Actor2CountryCode,
                   ActionGeo_FullName, ActionGeo_Type, NumMentions, AvgTone

    Returns
    -------
    pd.DataFrame
        컬럼: date, city, conflict_index, events, mentions, avg_tone
    """
    # Triad 필터
    mask = (
        df['Actor1CountryCode'].isin(TRIAD_COUNTRIES) &
        df['Actor2CountryCode'].isin(TRIAD_COUNTRIES) &
        df['EventCode'].astype(str).isin(CONFIRMED_CODES) &
        (df['ActionGeo_Type'] == 4)  # 도시 단위만
    )
    filtered = df[mask].copy()

    if filtered.empty:
        return pd.DataFrame(columns=['date', 'city', 'conflict_index',
                                      'events', 'mentions', 'avg_tone'])

    # 날짜 문자열 통일
    filtered['date'] = filtered['SQLDATE'].astype(str).str[:8]

    # W(AvgTone) 가중치 적용
    filtered['weight'] = filtered['AvgTone'].apply(tone_weight)
    filtered['weighted_mention'] = filtered['NumMentions'] * filtered['weight']

    # 도시별 일별 집계
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


# ──────────────────────────────────────────────
# 2. 칼만 필터 — Innovation(예측 오차) 추출
# ──────────────────────────────────────────────
def kalman_innovation(signal: np.ndarray,
                      Q: float = None,
                      R: float = None) -> dict:
    """
    1차원 칼만 필터를 적용하여 Innovation(예측 오차)을 추출.

    Parameters
    ----------
    signal : 1-d array
        도시의 일별 갈등지수 시계열
    Q, R : float, optional
        프로세스/관측 노이즈. None이면 초기 30일 분산에서 자동 추정.

    Returns
    -------
    dict with keys:
        'estimate'   : 칼만 추정값 (smoothed)
        'innovation' : 예측 오차 (= 실제 - 예측)
        'norm_innov' : 정규화된 Innovation (innovation / √S)
    """
    n = len(signal)
    if n < 2:
        return {
            'estimate': signal.copy(),
            'innovation': np.zeros(n),
            'norm_innov': np.zeros(n),
        }

    # 파라미터 자동 추정
    init_window = min(30, n)
    init_var = max(np.var(signal[:init_window]), 1e-6)

    if Q is None:
        Q = init_var * KALMAN_Q_RATIO
    if R is None:
        R = init_var * KALMAN_R_RATIO

    # 배열 초기화
    x_est = np.zeros(n)      # 추정 상태
    P_est = np.zeros(n)      # 불확실성
    innovation = np.zeros(n) # 예측 오차
    norm_innov = np.zeros(n) # 정규화 Innovation

    x_est[0] = signal[0]
    P_est[0] = R * KALMAN_P0_RATIO

    for t in range(1, n):
        # 예측(Predict)
        x_pred = x_est[t - 1]
        P_pred = P_est[t - 1] + Q

        # Innovation
        innovation[t] = signal[t] - x_pred
        S = P_pred + R  # Innovation 분산

        # 정규화 Innovation (0으로 나누기 방지)
        norm_innov[t] = innovation[t] / max(np.sqrt(S), 1e-10)

        # 업데이트(Update)
        K = P_pred / S  # 칼만 이득
        x_est[t] = x_pred + K * innovation[t]
        P_est[t] = (1 - K) * P_pred

    return {
        'estimate': x_est,
        'innovation': innovation,
        'norm_innov': norm_innov,
    }


# ──────────────────────────────────────────────
# 3. Rolling Z-score on Innovation
# ──────────────────────────────────────────────
def rolling_zscore(innovation: np.ndarray,
                   window: int = ROLLING_WINDOW) -> np.ndarray:
    """
    Innovation 시계열에 Rolling Window Z-score 적용.

    Parameters
    ----------
    innovation : 1-d array
        칼만 필터의 Innovation 출력
    window : int
        Rolling window 크기 (일)

    Returns
    -------
    1-d array : Innovation Z-score
    """
    n = len(innovation)
    z = np.zeros(n)

    for t in range(window, n):
        w = innovation[t - window:t]
        mean = w.mean()
        std = w.std()
        if std < 1e-10:
            z[t] = 0.0
        else:
            z[t] = (innovation[t] - mean) / std

    return z


# ──────────────────────────────────────────────
# 4. 전체 파이프라인: 이상 도시 목록 산출
# ──────────────────────────────────────────────
def detect_anomalies(city_daily: pd.DataFrame,
                     threshold: float = Z_THRESHOLD,
                     target_date: str = None) -> pd.DataFrame:
    """
    도시별 일별 갈등지수 DataFrame → 이상 징후 도시 목록.

    Parameters
    ----------
    city_daily : pd.DataFrame
        compute_conflict_index()의 출력
    threshold : float
        Z-score 임계치
    target_date : str, optional
        특정 날짜만 결과 반환. None이면 전체 기간.

    Returns
    -------
    pd.DataFrame
        컬럼: date, city, conflict_index, kalman_est, innovation,
              innov_z, events, mentions, avg_tone, is_anomaly
    """
    results = []

    for city, group in city_daily.groupby('city'):
        group = group.sort_values('date').reset_index(drop=True)
        signal = group['conflict_index'].values.astype(float)

        # 데이터가 너무 적으면 스킵
        if len(signal) < MIN_HISTORY:
            continue

        # 칼만 필터 → Innovation
        kf = kalman_innovation(signal)
        innovation = kf['innovation']

        # Rolling Z-score on Innovation
        iz = rolling_zscore(innovation)

        # 결과 조립
        group = group.copy()
        group['kalman_est'] = kf['estimate']
        group['innovation'] = innovation
        group['innov_z'] = iz
        group['is_anomaly'] = iz > threshold

        results.append(group)

    if not results:
        return pd.DataFrame()

    all_results = pd.concat(results, ignore_index=True)

    # 특정 날짜 필터
    if target_date:
        all_results = all_results[all_results['date'] == target_date]

    return all_results.sort_values(['date', 'innov_z'], ascending=[True, False])
