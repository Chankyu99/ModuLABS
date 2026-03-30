"""
찬규 담당 과제 v2: C안 core + broad 결합 기반 파이프라인 파라미터 산출
───────────────────────────────────────────────────────────────────────
수정사항:
  - Root 15(군사 태세)를 broad에서 추출하여 전이 행렬 & 골든타임 보완
  - 휘발성 임계치를 1.5σ로 조정하고 피크 감지 기준 재설계
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

BASE = Path("/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA")
CORE = BASE / "outputs" / "scenario_C" / "scenario_c_tasking_candidate_core.parquet"
COMBAT = BASE / "outputs" / "scenario_C" / "scenario_c_combat.parquet"
BROAD = BASE / "outputs" / "scenario_C" / "scenario_c_broad.parquet"
OUT = BASE / "outputs" / "pipeline_params"
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 0. 데이터 로드
# ─────────────────────────────────────────────
print("Loading datasets...")
df_core = pd.read_parquet(CORE)
df_core['dt'] = pd.to_datetime(df_core['SQLDATE'], format='%Y%m%d')
df_core['Root'] = df_core['EventRootCode'].astype(str).str.zfill(2)

# broad에서 Root 15 (군사 태세) 추출 — 전이 행렬 & 골든타임에 필요
df_broad = pd.read_parquet(BROAD)
df_broad['dt'] = pd.to_datetime(df_broad['SQLDATE'], format='%Y%m%d')
df_broad['Root'] = df_broad['EventRootCode'].astype(str).str.zfill(2)

# C안 핀셋 보정 로직 재적용하여 Root 15 이벤트도 동일 기준으로 필터링
PINSET_LOCAL_GEOS = ['gaza', 'jerusalem', 'hebron', 'ramallah', 'jenin', 'bethlehem', 'nablus', 'west bank']
TRIAD_ACTORS = ['IRN', 'USA']

df_root15 = df_broad[
    (df_broad['Root'] == '15') &
    (df_broad['IsRootEvent'] == 1) &
    (df_broad['GoldsteinScale'] < -7) &
    (df_broad['ActionGeo_Type'] == 4)
].copy()

# 핀셋 보정: 로컬 충돌 중 IRN/USA 비연결 건 제외
is_local_geo = df_root15['ActionGeo_FullName'].str.lower().str.contains('|'.join(PINSET_LOCAL_GEOS), na=False)
is_triad = (df_root15['Actor1CountryCode'].isin(TRIAD_ACTORS)) | (df_root15['Actor2CountryCode'].isin(TRIAD_ACTORS))
df_root15_core = df_root15[~(is_local_geo & ~is_triad)]

# Core + Root15 통합 데이터셋
df_full = pd.concat([df_core, df_root15_core], ignore_index=True).drop_duplicates(subset='GLOBALEVENTID')

print(f"  Core (18/19/20): {len(df_core):,} | Root15 추가: {len(df_root15_core):,}")
print(f"  통합 데이터셋: {len(df_full):,} rows\n")

# TOP5 지역 (core 기준)
top5_cities = df_core['ActionGeo_FullName'].value_counts().head(5).index.tolist()


# ══════════════════════════════════════════════
# TASK 1: 전이 행렬 수치 공식 문서화
# ══════════════════════════════════════════════
print("="*60)
print("TASK 1: Markov Transition Matrix (Root 15 포함)")
print("="*60)

# 지역별 시간순 정렬 후 전이 시퀀스 구성
df_seq = df_full.sort_values(['ActionGeo_FullName', 'dt', 'GLOBALEVENTID'])
df_seq['prev_root'] = df_seq.groupby('ActionGeo_FullName')['Root'].shift(1)
df_seq = df_seq.dropna(subset=['prev_root'])

# 전이 행렬
transition_counts = pd.crosstab(df_seq['prev_root'], df_seq['Root'])
transition_probs = pd.crosstab(df_seq['prev_root'], df_seq['Root'], normalize='index')

root_labels = {
    '15': '군사 태세 (Military Posture)',
    '18': '폭행/강제 (Assault)',
    '19': '무력 사용 (Use of Force)',
    '20': '대량 폭력 (Mass Violence)'
}

# 저장
transition_probs.to_csv(OUT / "transition_matrix_probs.csv")
transition_counts.to_csv(OUT / "transition_matrix_counts.csv")

# 핵심 전이 경로 추출
critical_paths = []
for prev in transition_probs.index:
    for curr in transition_probs.columns:
        prob = transition_probs.loc[prev, curr]
        count = transition_counts.loc[prev, curr]
        if prob >= 0.05:
            if prob >= 0.5:
                priority = 'CRITICAL'
            elif prob >= 0.3:
                priority = 'HIGH'
            elif prob >= 0.1:
                priority = 'MEDIUM'
            else:
                priority = 'LOW'

            critical_paths.append({
                'from_code': prev,
                'from_label': root_labels.get(prev, prev),
                'to_code': curr,
                'to_label': root_labels.get(curr, curr),
                'probability': round(prob, 4),
                'count': int(count),
                'trigger_priority': priority,
                'action': '즉시 촬영 예약' if priority == 'CRITICAL' else
                          '대기 모드 전환' if priority == 'HIGH' else
                          '모니터링 강화' if priority == 'MEDIUM' else '일반 관찰'
            })

critical_df = pd.DataFrame(critical_paths).sort_values('probability', ascending=False)
critical_df.to_csv(OUT / "critical_transition_paths.csv", index=False)

print("\n[전이 확률 행렬]")
print(transition_probs.round(4).to_string())
print(f"\n핵심 전이 경로 {len(critical_df)}개:")
for _, row in critical_df.iterrows():
    print(f"  Root {row['from_code']}→{row['to_code']}: {row['probability']:.4f} ({row['trigger_priority']}) → {row['action']}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sns.heatmap(transition_probs, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0],
            xticklabels=[f"Root {c}" for c in transition_probs.columns],
            yticklabels=[f"Root {c}" for c in transition_probs.index])
axes[0].set_title('Transition Probability (C-plan, Root15 incl.)')
axes[0].set_xlabel('Next State')
axes[0].set_ylabel('Current State')

sns.heatmap(transition_counts, annot=True, fmt=',d', cmap='Blues', ax=axes[1],
            xticklabels=[f"Root {c}" for c in transition_counts.columns],
            yticklabels=[f"Root {c}" for c in transition_counts.index])
axes[1].set_title('Transition Counts')
axes[1].set_xlabel('Next State')
axes[1].set_ylabel('Current State')
plt.tight_layout()
plt.savefig(OUT / "transition_matrix_visual.png", dpi=150)


# ══════════════════════════════════════════════
# TASK 2: TOP5 지역 감성 휘발성 임계치 수치화
# ══════════════════════════════════════════════
print("\n" + "="*60)
print("TASK 2: TOP5 Volatility Threshold (Rolling Baseline)")
print("="*60)

N_VALUES = [4, 8]
SIGMA_MULTIPLIERS = [1.5, 2.0]

volatility_results = []
peak_detection_scores = {}

fig, axes = plt.subplots(len(top5_cities), 1, figsize=(18, 5 * len(top5_cities)))

for idx, city in enumerate(top5_cities):
    city_df = df_full[df_full['ActionGeo_FullName'] == city].copy()

    # 주간 통계
    weekly = city_df.groupby(pd.Grouper(key='dt', freq='W')).agg(
        tone_std=('AvgTone', 'std'),
        tone_mean=('AvgTone', 'mean'),
        event_count=('GLOBALEVENTID', 'count'),
        mention_sum=('NumMentions', 'sum'),
        goldstein_min=('GoldsteinScale', 'min')
    ).dropna(subset=['tone_std'])
    weekly = weekly[weekly['event_count'] >= 3]

    ax = axes[idx]
    ax.plot(weekly.index, weekly['tone_std'], color='gray', alpha=0.4, linewidth=1, label='Raw Volatility')

    best_config = {'n': None, 'sigma': None, 'f1': 0}

    for n in N_VALUES:
        for sigma_mult in SIGMA_MULTIPLIERS:
            baseline = weekly['tone_std'].rolling(window=n, min_periods=max(2, n//2)).mean()
            baseline_std = weekly['tone_std'].rolling(window=n, min_periods=max(2, n//2)).std()
            threshold = baseline + sigma_mult * baseline_std
            is_peak = weekly['tone_std'] > threshold

            # 위기 판별 기준: 동시에 (1) 멘션 폭증 OR (2) 골드스타인 극저
            mention_p90 = weekly['mention_sum'].quantile(0.85)
            goldstein_extreme = weekly['goldstein_min'].quantile(0.15)
            actual_crisis = (weekly['mention_sum'] >= mention_p90) | (weekly['goldstein_min'] <= goldstein_extreme)

            tp = (is_peak & actual_crisis).sum()
            fp = (is_peak & ~actual_crisis).sum()
            fn = (~is_peak & actual_crisis).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            peak_weeks = weekly[is_peak]
            avg_mult = (peak_weeks['tone_std'] / baseline[is_peak]).mean() if not peak_weeks.empty and baseline[is_peak].mean() > 0 else 0

            volatility_results.append({
                'city': city,
                'rolling_N': n,
                'sigma_mult': sigma_mult,
                'total_weeks': len(weekly),
                'peaks_detected': int(is_peak.sum()),
                'peak_ratio': round(is_peak.mean(), 4),
                'avg_baseline_multiplier': round(avg_mult, 2),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4)
            })

            if f1 > best_config['f1']:
                best_config = {'n': n, 'sigma': sigma_mult, 'f1': f1}

    # 최적 설정으로 시각화
    if best_config['n']:
        bl = weekly['tone_std'].rolling(window=best_config['n'], min_periods=2).mean()
        bs = weekly['tone_std'].rolling(window=best_config['n'], min_periods=2).std()
        thresh = bl + best_config['sigma'] * bs
        peaks = weekly['tone_std'] > thresh

        ax.plot(weekly.index, bl, color='blue', alpha=0.6, label=f"Baseline (N={best_config['n']}w)")
        ax.plot(weekly.index, thresh, color='red', linestyle='--', alpha=0.6,
                label=f"Threshold ({best_config['sigma']}σ)")
        ax.fill_between(weekly.index, weekly['tone_std'], thresh,
                        where=peaks, color='red', alpha=0.3, label='PEAK ALERT')

    peak_detection_scores[city] = best_config
    ax.set_title(f"{city} — Best: N={best_config['n']}w, {best_config['sigma']}σ (F1={best_config['f1']:.3f})")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "top5_volatility_thresholds.png", dpi=150)

vol_df = pd.DataFrame(volatility_results)
vol_df.to_csv(OUT / "volatility_threshold_analysis.csv", index=False)

optimal_summary = pd.DataFrame([
    {'city': c, 'optimal_N': v['n'], 'optimal_sigma': v['sigma'], 'f1': round(v['f1'], 4)}
    for c, v in peak_detection_scores.items()
])
optimal_summary.to_csv(OUT / "optimal_rolling_window.csv", index=False)

print("\n[최적 설정 요약]")
print(optimal_summary.to_string(index=False))


# ══════════════════════════════════════════════
# TASK 3: 골든타임 & 스케줄링 파라미터 통합
# ══════════════════════════════════════════════
print("\n" + "="*60)
print("TASK 3: Golden Time + Scheduling Parameters")
print("="*60)

# 골든타임 산출 (Root 15 → Root 18/19/20)
signs = df_full[df_full['Root'] == '15']
incidents = df_full[df_full['Root'].isin(['18', '19', '20'])]

leads_by_city = {}
for city in top5_cities:
    city_signs = signs[signs['ActionGeo_FullName'] == city]
    city_incs = incidents[incidents['ActionGeo_FullName'] == city]
    leads = []
    for _, s in city_signs.iterrows():
        f_inc = city_incs[
            (city_incs['dt'] > s['dt']) &
            (city_incs['dt'] <= s['dt'] + pd.Timedelta(days=14))
        ]
        if not f_inc.empty:
            leads.append((f_inc.iloc[0]['dt'] - s['dt']).days)
    leads_by_city[city] = leads
    print(f"  {city}: {len(leads)} sign→incident 시퀀스, avg={np.mean(leads):.1f}d" if leads else f"  {city}: 시퀀스 없음")

# 통합 파라미터 테이블
scheduling_params = []
for city in top5_cities:
    leads = leads_by_city.get(city, [])
    opt = peak_detection_scores.get(city, {})
    city_data = df_core[df_core['ActionGeo_FullName'] == city]

    p_19_19 = transition_probs.loc['19', '19'] if '19' in transition_probs.index else 0
    p_15_19 = transition_probs.loc['15', '19'] if '15' in transition_probs.index and '19' in transition_probs.columns else 0
    p_18_19 = transition_probs.loc['18', '19'] if '18' in transition_probs.index and '19' in transition_probs.columns else 0

    scheduling_params.append({
        'city': city,
        'lat': city_data['ActionGeo_Lat'].mode().iloc[0] if len(city_data) > 0 else None,
        'lon': city_data['ActionGeo_Long'].mode().iloc[0] if len(city_data) > 0 else None,
        'event_count': len(city_data),
        'avg_lead_time_days': round(np.mean(leads), 2) if leads else None,
        'median_lead_time_days': round(np.median(leads), 2) if leads else None,
        'within_3days_pct': round((np.array(leads) <= 3).mean() * 100, 1) if leads else None,
        'vol_window_N': opt.get('n'),
        'vol_sigma': opt.get('sigma'),
        'vol_f1': round(opt.get('f1', 0), 4),
        'p_15to19': round(p_15_19, 4),
        'p_18to19': round(p_18_19, 4),
        'p_19to19': round(p_19_19, 4),
    })

params_df = pd.DataFrame(scheduling_params)
params_df.to_csv(OUT / "scheduling_parameters.csv", index=False)

# JSON 파이프라인 연동 포맷
params_json = {
    "version": "v0.2",
    "data_source": "scenario_c_tasking_candidate_core.parquet + root15 from broad",
    "transition_matrix": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in transition_probs.to_dict().items()},
    "critical_transitions": critical_df.to_dict(orient='records'),
    "scheduling_targets": params_df.to_dict(orient='records'),
    "golden_time": {
        "recommended_window_days": 3,
        "rationale": "1~3일 내 사건 집중 발생 구간"
    },
    "trigger_conditions": {
        "markov_19to19": round(float(transition_probs.loc['19', '19']), 4) if '19' in transition_probs.index else None,
        "markov_15to19": round(float(transition_probs.loc['15', '19']), 4) if '15' in transition_probs.index and '19' in transition_probs.columns else None,
        "markov_18to19": round(float(transition_probs.loc['18', '19']), 4) if '18' in transition_probs.index and '19' in transition_probs.columns else None,
        "volatility_sigma_threshold": 1.5,
        "tone_crisis_threshold": -10.0
    }
}

with open(OUT / "pipeline_params.json", 'w', encoding='utf-8') as f:
    json.dump(params_json, f, ensure_ascii=False, indent=2, default=str)

print("\n[최종 스케줄링 파라미터 통합 테이블]")
print(params_df.to_string(index=False))

print(f"\n{'='*60}")
print(f"모든 산출물 → {OUT}")
for f_item in sorted(OUT.iterdir()):
    print(f"  - {f_item.name} ({f_item.stat().st_size / 1024:.1f} KB)")
