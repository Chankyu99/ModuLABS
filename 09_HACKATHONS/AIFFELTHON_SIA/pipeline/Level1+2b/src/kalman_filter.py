import numpy as np
import pandas as pd
from numba import jit

from src.preprocess import clean_gdelt_data, apply_standard_scaling
from src.config import (
    KALMAN_Q, KALMAN_R, KALMAN_Q_RATIO, KALMAN_R_RATIO, KALMAN_P0_RATIO, KALMAN_MIN_VAR,
    MIN_HISTORY, RISK_THRESHOLDS, RISK_LEVELS_LIST, 
    RISK_LABELS_LIST, RISK_GUIDES_LIST, RISK_EMOJIS_LIST,
    COEFS
)

# ─── 1. 칼만 필터 코어 (Numba JIT) ───────────────────────

@jit(nopython=True)
def kalman_innovation(signal: np.ndarray, manual_q=-1.0, manual_r=-1.0) -> tuple:
    """
    수동 설정값(manual_q, manual_r)이 -1.0이면(기본값) 30일 분산 기반 자동 추정을 수행합니다.
    """
    n = len(signal)
                
    # 초기 30일 데이터를 기준으로 노이즈(Q, R) 자동 추정
    init_window = min(30, n)
    init_var = max(np.var(signal[:init_window]), KALMAN_MIN_VAR)
    
    # 수동 설정값이 있으면 사용, 없으면 자동 계산
    Q = manual_q if manual_q > 0 else init_var * KALMAN_Q_RATIO
    R = manual_r if manual_r > 0 else init_var * KALMAN_R_RATIO

    x_est = np.zeros(n)      
    P_est = np.zeros(n)      
    innovation = np.zeros(n) 
    norm_innov = np.zeros(n) 

    x_est[0] = signal[0]
    P_est[0] = R * KALMAN_P0_RATIO

    for t in range(1, n):
        x_pred = x_est[t - 1]
        P_pred = P_est[t - 1] + Q

        # Kalman Innovation
        innovation[t] = signal[t] - x_pred
        S = P_pred + R
        
        # 표준화
        norm_innov[t] = innovation[t] / max(np.sqrt(S), 1e-10)

        K = P_pred / S 
        x_est[t] = x_pred + K * innovation[t]
        P_est[t] = (1 - K) * P_pred

    return x_est, innovation, norm_innov


# ─── 2. 도시별 시계열 그룹화 처리 ─────────────────────────

def apply_kalman_group(group, global_min_date, global_max_date, manual_q: float = -1.0, manual_r: float = -1.0):
    """도시별 첫 등장일부터의 시계열 생성 및 필터 적용. manual_q/manual_r > 0이면 Q/R 수동 주입."""
    if group.empty: return None

    # Pandas GroupBy 특징을 이용해 ID와 지명 추출
    fid = group.name if hasattr(group, 'name') else group['ActionGeo_FeatureID'].iloc[0]
    city_name = group['city'].iloc[0]

    group = group.sort_values('date')
    all_dates = pd.date_range(start=global_min_date, end=global_max_date, freq='D')

    # 뼈대 생성 및 숫자 컬럼 0 채우기
    group = group.set_index('date').reindex(all_dates).reset_index().rename(columns={'index': 'date'})
    num_cols = ['conflict_index', 'events', 'mentions', 'articles', 'sources', 'avg_tone']
    group[num_cols] = group[num_cols].fillna(0)

    # 복구
    group['ActionGeo_FeatureID'] = fid
    group['city'] = city_name

    signal = np.ascontiguousarray(group['conflict_index'].values, dtype=np.float64)
    if len(signal) < MIN_HISTORY: return None

    # Core 연산 호출 (Q/R 주입)
    x_est, innov, norm_innov = kalman_innovation(signal, manual_q, manual_r)
    
    group['kalman_est'] = x_est
    group['innovation'] = innov
    group['innov_z'] = norm_innov

    # 리스크 등급 분류
    z_scores = group['innov_z'].values
    conditions = [z_scores >= t for t in RISK_THRESHOLDS]
    group['risk_level'] = np.select(conditions, RISK_LEVELS_LIST, default=0)
    group['risk_label'] = np.select(conditions, RISK_LABELS_LIST, default='정상')
    group['risk_guide'] = np.select(conditions, RISK_GUIDES_LIST, default='평시 수준 유지')
    group['risk_emoji'] = np.select(conditions, RISK_EMOJIS_LIST, default='🔵')

    group['is_anomaly'] = group['risk_level'] >= 1
    group['date'] = group['date'].dt.strftime('%Y%m%d')
    
    return group


# ─── 3. 메인 파이프라인 함수 ───────────────────────────

def compute_conflict_index(df: pd.DataFrame, is_train: bool = False, manual_q: float = KALMAN_Q, manual_r: float = KALMAN_R) -> tuple:
    """
    is_train=True: 스케일러 학습 및 저장 모드
    is_train=False: 저장된 스케일러 로드 모드
    """
    # 1. 기본 정제 (국가, 코드 필터링 등)
    filtered = clean_gdelt_data(df)
    if filtered.empty: 
        return pd.DataFrame(), filtered

    # 2. 'date' 컬럼 보장 (KeyError 방지)
    if 'date' not in filtered.columns:
        if 'SQLDATE' in filtered.columns:
            filtered['date'] = filtered['SQLDATE'].astype(str).str[:8]
        else:
            raise KeyError("데이터에 'date' 또는 'SQLDATE' 컬럼이 존재하지 않습니다.")

    # 3. 스케일링 적용 (Log_Mentions, AvgTone_Sq 자동 생성 및 Z-Score 변환)
    # src.preprocess에서 가져온 apply_standard_scaling 함수 사용
    filtered = apply_standard_scaling(filtered, is_train=is_train)
    
    # 4. 황금 가중치 공식 적용 (Z_t 산출)
    # config.py의 COEFS 이름을 확인하세요 (이전 가이드에선 FINAL_COEFS로 명명)
    filtered['weighted_index'] = (
        (COEFS['w_log_mentions'] * filtered['Log_Mentions']) +
        (COEFS['w_log_sources'] * filtered['Log_Sources']) +
        (COEFS['w_goldstein'] * filtered['GoldsteinScale']) +
        (COEFS['w_avgtone'] * filtered['AvgTone']) +
        (COEFS['w_avgtone_sq'] * filtered['AvgTone_Sq'])
    )

    # 5. 도시별 집계 (스케일링된 지수 기반)
    agg = (filtered.groupby(['date', 'ActionGeo_FeatureID', 'ActionGeo_FullName'])
            .agg(
                conflict_index=('weighted_index', 'sum'),
                events=('EventCode', 'count'),            # 사건 수
                mentions=('NumMentions', 'mean'),         # 언급량 평균
                articles=('NumArticles', 'mean'),         # 기사 수 평균
                sources=('NumSources', 'mean'),           # 출처 수 평균
                avg_tone=('AvgTone', 'mean'),             # 평균 톤
                avg_goldstein=('GoldsteinScale', 'mean'), # 평균 골드스타인 지수
                country_code=('ActionGeo_CountryCode', 'first'),
                lat=('ActionGeo_Lat', 'first'),           # 합쳐진 그룹 중 첫 번째 좌표 선택
                lng=('ActionGeo_Long', 'first') 
            ).reset_index().rename(columns={'ActionGeo_FullName': 'city'}))

    # 6. 날짜 형식 변환 및 범위 확정
    agg['date'] = pd.to_datetime(agg['date'])
    global_min_date = agg['date'].min()
    global_max_date = agg['date'].max()

    # 7. 도시별 그룹화 및 칼만 필터(이상 탐지) 호출
    results_df = agg.groupby('ActionGeo_FeatureID', group_keys=False).apply(
        lambda x: apply_kalman_group(x, global_min_date, global_max_date, manual_q, manual_r)
    )
    
    if results_df is None or results_df.empty: 
        return pd.DataFrame(), filtered

    # 최종 결과 반환 (결측치 제거 및 인덱스 초기화)
    return results_df.dropna(subset=['innov_z']).reset_index(drop=True), filtered


def detect_anomalies(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    if df.empty: return df
    return df[(df['date'] == str(target_date)) & (df['is_anomaly'] == True)].copy()
