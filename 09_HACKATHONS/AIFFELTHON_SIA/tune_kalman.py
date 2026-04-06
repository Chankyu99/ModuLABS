"""
칼만 필터 하이퍼파라미터 그리드 서치
─────────────────────────────────
Ground Truth:
  - 2/28: Minab가 이상 징후(Z >= 5.0, YELLOW 이상)로 탐지되어야 함
  - 3/31: Isfahan, Beirut, Baghdad, Tabriz, Kharg Island가 Top 5에 포함되어야 함
"""
import numpy as np
import pandas as pd
import itertools
import sys

from pipeline.gdelt_fetcher import load_all_data
from pipeline.conflict_index import compute_conflict_index
from pipeline.config import CONFIRMED_CODES, MONITORED_COUNTRIES

import warnings
warnings.filterwarnings('ignore')

def kalman_with_params(signal, Q_ratio, R_ratio, P0_ratio):
    """칼만 필터를 커스텀 파라미터로 실행"""
    n = len(signal)
    if n < 2:
        return np.zeros(n)
    
    init_window = min(30, n)
    init_var = max(np.var(signal[:init_window]), 1e-6)
    
    Q = init_var * Q_ratio
    R = init_var * R_ratio
    
    x_est = np.zeros(n)
    P_est = np.zeros(n)
    norm_innov = np.zeros(n)
    
    x_est[0] = signal[0]
    P_est[0] = R * P0_ratio
    
    for t in range(1, n):
        x_pred = x_est[t-1]
        P_pred = P_est[t-1] + Q
        
        innovation = signal[t] - x_pred
        S = P_pred + R
        norm_innov[t] = innovation / max(np.sqrt(S), 1e-10)
        
        K = P_pred / S
        x_est[t] = x_pred + K * innovation
        P_est[t] = (1 - K) * P_pred
    
    return norm_innov

# ── 데이터 로드 ──
print("Loading Data...")
df = load_all_data(include_daily=True)
city_daily = compute_conflict_index(df)

# ── Ground Truth 정의 ──
GT_0228 = {'date': '20260228', 'must_detect': ['Minab']}
GT_0331 = {'date': '20260331', 'must_detect': ['Isfahan', 'Beirut', 'Baghdad', 'Tabriz', 'Kharg Island']}

# ── 그리드 서치 범위 ──
Q_ratios = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
R_ratios = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
P0_ratios = [1.0, 1.5, 2.0, 3.0, 5.0]

YELLOW_THRESHOLD = 5.0

print(f"Testing {len(Q_ratios) * len(R_ratios) * len(P0_ratios)} parameter combinations...")
print()

results = []

for Q_r, R_r, P0_r in itertools.product(Q_ratios, R_ratios, P0_ratios):
    scores = {}
    
    for gt in [GT_0228, GT_0331]:
        target_date = gt['date']
        
        for city, group in city_daily.groupby('city'):
            g_sorted = group.sort_values('date')
            g_target = g_sorted[g_sorted['date'] <= target_date]
            if len(g_target) < 30:
                continue
            if g_target.iloc[-1]['date'] != target_date:
                continue
            
            signal = g_target['conflict_index'].values.astype(float)
            z_scores = kalman_with_params(signal, Q_r, R_r, P0_r)
            z_final = z_scores[-1]
            
            key = (target_date, city)
            scores[key] = z_final
    
    # 검증 1: 2/28 Minab 이상 탐지 여부
    minab_detected = False
    for (date, city), z in scores.items():
        if date == '20260228' and 'Minab' in city and z >= YELLOW_THRESHOLD:
            minab_detected = True
            minab_z = z
            break
    
    # 검증 2: 3/31 Top 5 명중률
    scores_0331 = {city: z for (date, city), z in scores.items() if date == '20260331'}
    top5_cities = sorted(scores_0331, key=scores_0331.get, reverse=True)[:5]
    top5_names = [c.split(',')[0].strip() for c in top5_cities]
    
    hits = sum(1 for target in GT_0331['must_detect'] if any(target in name for name in top5_names))
    
    if minab_detected and hits == 5:
        minab_z_val = next((z for (d, c), z in scores.items() if d == '20260228' and 'Minab' in c), 0)
        top1_z = scores_0331[top5_cities[0]] if top5_cities else 0
        
        results.append({
            'Q': Q_r, 'R': R_r, 'P0': P0_r,
            'Minab_Z': round(minab_z_val, 2),
            'Top5_Hits': hits,
            'Top1': top5_names[0] if top5_names else '',
            'Top1_Z': round(top1_z, 2),
            'Top5': top5_names
        })

print(f"\n{'='*80}")
print(f"  ✅ 두 조건 모두 만족하는 파라미터 조합: {len(results)}개")
print(f"{'='*80}")

if results:
    for i, r in enumerate(results[:10], 1):
        print(f"\n  [{i}] Q={r['Q']}, R={r['R']}, P0={r['P0']}")
        print(f"      Minab 2/28 Z = {r['Minab_Z']}")
        print(f"      3/31 Top5 = {r['Top5']} (1위 Z={r['Top1_Z']})")
    
    # 현재 파라미터와 비교
    print(f"\n{'─'*80}")
    print(f"  현재 파라미터: Q=0.01, R=1.0, P0=2.0")
    current = [r for r in results if r['Q'] == 0.01 and r['R'] == 1.0 and r['P0'] == 2.0]
    if current:
        print(f"  → 현재 파라미터도 두 조건을 만족합니다!")
    else:
        print(f"  → 현재 파라미터는 두 조건을 동시에 만족하지 못합니다.")
        
        # 현재 파라미터로 Minab 확인
        for (d, c), z in scores.items():
            if d == '20260228' and 'Minab' in c:
                print(f"  → 현재 Minab 2/28 Z = {z:.2f} (임계치 {YELLOW_THRESHOLD})")
else:
    print("  ❌ 두 조건을 동시에 만족하는 조합이 없습니다.")
    print("  tone_weight 가중치 또는 리스크 임계치 조정이 필요할 수 있습니다.")
    
    # 각 조건별 최선 출력
    print(f"\n  [참고] 현재 파라미터(Q=0.01, R=1.0, P0=2.0)로 체크:")
    scores_current = {}
    for gt in [GT_0228, GT_0331]:
        for city, group in city_daily.groupby('city'):
            g_sorted = group.sort_values('date')
            g_target = g_sorted[g_sorted['date'] <= gt['date']]
            if len(g_target) < 30 or g_target.iloc[-1]['date'] != gt['date']:
                continue
            signal = g_target['conflict_index'].values.astype(float)
            z_scores = kalman_with_params(signal, 0.01, 1.0, 2.0)
            scores_current[(gt['date'], city)] = z_scores[-1]
    
    for (d, c), z in sorted(scores_current.items(), key=lambda x: x[1], reverse=True):
        if d == '20260228' and 'Minab' in c:
            print(f"  Minab 2/28: Z = {z:.4f}")
