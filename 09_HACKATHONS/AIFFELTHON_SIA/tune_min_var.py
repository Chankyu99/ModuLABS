"""
MIN_INIT_VAR 최적값 탐색
Ground Truth:
  2/28: Minab ≥ YELLOW (Z≥5)
  3/31: Isfahan, Beirut, Baghdad, Tabriz, Kharg Island 모두 Top 5
"""
import numpy as np
import pandas as pd
import pipeline.config as cfg
from pipeline.gdelt_fetcher import load_all_data
from pipeline.conflict_index import compute_conflict_index, kalman_innovation

import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
raw = load_all_data()

# MIN_INIT_VAR을 고정하지 않고, compute_conflict_index는 한 번만 호출
city_daily = compute_conflict_index(raw)

GT_CITIES = ['Isfahan', 'Beirut', 'Baghdad', 'Tabriz', 'Kharg Island']
test_vars = [1, 5, 10, 25, 50, 100, 200, 500, 1000]

print(f"\n{'MIN_VAR':>8s} | {'Minab Z':>10s} | {'Minab등급':>8s} | Top5 명중 | Top5 도시 및 Z-Score")
print(f"{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*50}")

for min_var in test_vars:
    scores_228 = {}
    scores_331 = {}
    
    for city, group in city_daily.groupby('city'):
        g = group.sort_values('date').reset_index(drop=True)
        signal = g['conflict_index'].values.astype(float)
        dates = g['date'].values
        
        if len(signal) < 30:
            continue
        
        # 칼만 필터 직접 실행 (min_var 적용)
        n = len(signal)
        init_window = min(30, n)
        init_var = max(np.var(signal[:init_window]), min_var)
        
        Q = init_var * cfg.KALMAN_Q_RATIO
        R = init_var * cfg.KALMAN_R_RATIO
        
        x_est = np.zeros(n)
        P_est = np.zeros(n)
        norm_innov = np.zeros(n)
        
        x_est[0] = signal[0]
        P_est[0] = R * cfg.KALMAN_P0_RATIO
        
        for t in range(1, n):
            x_pred = x_est[t-1]
            P_pred = P_est[t-1] + Q
            innov = signal[t] - x_pred
            S = P_pred + R
            norm_innov[t] = innov / max(np.sqrt(S), 1e-10)
            K = P_pred / S
            x_est[t] = x_pred + K * innov
            P_est[t] = (1 - K) * P_pred
        
        # 2/28
        if '20260228' in dates:
            idx = list(dates).index('20260228')
            scores_228[city] = norm_innov[idx]
        
        # 3/31
        if '20260331' in dates:
            idx = list(dates).index('20260331')
            scores_331[city] = norm_innov[idx]
    
    # Minab 2/28
    minab_z = 0
    minab_grade = '-'
    for c, z in scores_228.items():
        if 'Minab' in c:
            minab_z = z
            if z >= 150: minab_grade = '🛑RED'
            elif z >= 20: minab_grade = '🟠ORA'
            elif z >= 5: minab_grade = '🟡YEL'
            else: minab_grade = '🔵BLU'
    
    # 3/31 Top 5
    top5 = sorted(scores_331.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_names = [c.split(',')[0].strip() for c, _ in top5]
    hits = sum(1 for gt in GT_CITIES if any(gt in name for name in top5_names))
    
    top5_str = ", ".join([f"{c.split(',')[0][:10]}({z:.0f})" for c, z in top5])
    
    print(f"{min_var:>8.0f} | {minab_z:>10.1f} | {minab_grade:>8s} | {hits}/5      | {top5_str}")

print()
print("=" * 100)
print("평가 기준: Minab ≥ YELLOW(Z≥5) AND Top5 명중률 = 5/5")
