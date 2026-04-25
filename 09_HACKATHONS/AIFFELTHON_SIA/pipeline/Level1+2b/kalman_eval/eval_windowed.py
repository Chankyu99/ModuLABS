"""
사건 발생일로부터 +N일 이내 탐지를 정답으로 인정하는 windowed 평가.

GT date D → 탐지 인정 범위: [D, D+N]  (forward-only, 미리 찍는 건 인정 안 함)
Precision: 플래그된 anomaly date T → GT에 [T-N, T] 내에 해당 도시가 있으면 TP
  (즉, 이벤트 발생 후 N일 이내에 탐지된 경우만 TP)

기존 run 로그(verified.csv)를 그대로 읽어 재평가 — LLM 재실행 없음.

Tier stratification:
  gt_detectability.csv 로드 → tier A (mentions≥10) / C (mentions<10) 분리 recall

Usage:
  python eval_windowed.py                         # window 0/3/5 비교
  python eval_windowed.py --run logs/runs/<dir>   # 특정 run 지정
  python eval_windowed.py --window 3              # 단일 window
"""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
GT_DIR = PROJECT_ROOT / "data" / "ground_truth"
TIER_PATH = PROJECT_ROOT / "gt_detectability.csv"


def get_distance(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p)*np.cos(lat2*p)*(1-np.cos((lon2-lon1)*p))/2
    return 12742 * np.arcsin(np.sqrt(a))


def load_gt_all():
    rows = []
    for f in sorted(os.listdir(GT_DIR)):
        if not f.endswith(".csv"): continue
        df = pd.read_csv(GT_DIR / f)
        for _, r in df.iterrows():
            rows.append({
                'date': f[:8],
                'city': str(r['ActionGeo_FullName']).strip().lower(),
                'lat': r.get('Lat'), 'lon': r.get('Long'),
            })
    return pd.DataFrame(rows)


def load_tiers(threshold: int = None):
    """gt_detectability.csv에서 (date, city) → tier 매핑 로드.
    threshold 지정 시 mentions_on_date 기준으로 재계산."""
    if not TIER_PATH.exists():
        return {}
    df = pd.read_csv(TIER_PATH)
    df['date'] = df['date'].astype(str)
    df['city_l'] = df['city'].astype(str).str.lower().str.strip()
    # A = mentions≥threshold AND match_type ∈ {exact,alias,substring}
    # geo fallback은 이름이 GDELT에 없어 Kalman이 구조적으로 못 잡는 케이스 → A에서 제외
    valid_match = df['match_type'].isin(['exact', 'alias', 'substring'])
    if threshold is not None:
        df['tier'] = np.where((df['mentions_on_date'] >= threshold) & valid_match, 'A', 'C')
    else:
        df['tier'] = np.where(df['tier'].eq('A') & valid_match, 'A', 'C')
    return {(row['date'], row['city_l']): row['tier'] for _, row in df.iterrows()}


def date_range(start: str, days: int):
    dt = datetime.strptime(start, '%Y%m%d')
    return [(dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(days+1)]


def fbeta(p, r, beta):
    if p + r == 0: return 0.0
    b2 = beta ** 2
    return (1 + b2) * p * r / (b2 * p + r)


def evaluate_windowed(run_dir: Path, window: int = 0, radius_km: float = 50.0, beta: float = 1.5,
                      gt=None, verified=None, tiers=None):
    if gt is None:
        gt = load_gt_all()
    if tiers is None:
        tiers = load_tiers()

    if verified is None:
        dates = sorted([d for d in os.listdir(run_dir / "by_date")
                        if (run_dir / "by_date" / d).is_dir()])
        all_verified = []
        for d in dates:
            vf = run_dir / "by_date" / d / "verified.csv"
            if not vf.exists() or vf.stat().st_size < 10: continue
            df = pd.read_csv(vf)
            if df.empty: continue
            df['run_date'] = d
            all_verified.append(df)
        if not all_verified:
            print("No verified.csv found"); return None
        verified = pd.concat(all_verified, ignore_index=True)
        verified['city_l'] = verified['city'].astype(str).str.lower().str.strip()

    def gt_match_any(vrow, gt_subset):
        city = vrow['city_l']
        lat, lon = vrow.get('lat'), vrow.get('lng')
        for _, g in gt_subset.iterrows():
            gc = g['city']
            if gc and (gc == city or gc in city or city in gc):
                return g['city']
            if pd.notna(lat) and pd.notna(lon) and pd.notna(g['lat']) and pd.notna(g['lon']):
                if get_distance(float(lat), float(lon), float(g['lat']), float(g['lon'])) <= radius_km:
                    return g['city']
        return None

    # ── RECALL: GT(D, city) → detected on [D, D+window] ──
    hit_gt = set()
    for _, g in gt.iterrows():
        window_dates = date_range(g['date'], window)
        for wd in window_dates:
            vday = verified[verified['run_date'] == wd]
            if vday.empty: continue
            gc = g['city']
            name_hit = vday['city_l'].apply(
                lambda c: gc in c or c in gc or gc == c
            ).any()
            if name_hit:
                hit_gt.add((g['date'], g['city']))
                break
            if pd.notna(g['lat']) and pd.notna(g['lon']):
                for _, vr in vday.iterrows():
                    if pd.notna(vr.get('lat')) and pd.notna(vr.get('lng')):
                        if get_distance(float(g['lat']), float(g['lon']),
                                        float(vr['lat']), float(vr['lng'])) <= radius_km:
                            hit_gt.add((g['date'], g['city']))
                            break
                if (g['date'], g['city']) in hit_gt:
                    break

    n_gt = len(gt)
    recall = len(hit_gt) / n_gt if n_gt > 0 else 0.0

    # Tier-stratified recall
    gt_a = [(d, c) for d, c in [(r['date'], r['city']) for _, r in gt.iterrows()]
            if tiers.get((d, c), 'A') == 'A']
    gt_c = [(d, c) for d, c in [(r['date'], r['city']) for _, r in gt.iterrows()]
            if tiers.get((d, c), 'A') == 'C']
    hit_a = len([k for k in hit_gt if k in set(gt_a)])
    hit_c = len([k for k in hit_gt if k in set(gt_c)])
    recall_a = hit_a / len(gt_a) if gt_a else 0.0
    recall_c = hit_c / len(gt_c) if gt_c else 0.0

    # ── PRECISION: detection T → TP if GT event in [T-window, T] ──
    n_tp = n_tp_a = n_fp = n_excluded = 0
    TP_STATUSES = {'SUCCESS', 'AMBIGUOUS'}
    FP_STATUSES = {'DROPPED', 'NO_MENTION', 'DATE_MISMATCH'}
    gt_a_set = set(gt_a)

    for _, vr in verified.iterrows():
        status = vr.get('llm_status', '')
        run_d = vr['run_date']
        window_gt_dates = date_range(
            (datetime.strptime(run_d, '%Y%m%d') - timedelta(days=window)).strftime('%Y%m%d'),
            window
        )
        gt_window = gt[gt['date'].isin(window_gt_dates)]
        matched = gt_match_any(vr, gt_window) if not gt_window.empty else None

        if matched is not None:
            judgement = 'TP'
            # A-tier GT 매칭 여부 확인
            matched_key = next(
                ((d, c) for d, c in gt_window[['date','city']].itertuples(index=False)
                 if c and (c == vr['city_l'] or c in vr['city_l'] or vr['city_l'] in c)),
                None
            )
            if matched_key and matched_key in gt_a_set:
                n_tp_a += 1
        elif status in TP_STATUSES:
            judgement = 'TP'
        elif status in FP_STATUSES:
            judgement = 'FP'
        else:
            judgement = 'EXCLUDED'

        if judgement == 'TP': n_tp += 1
        elif judgement == 'FP': n_fp += 1
        else: n_excluded += 1

    denom = n_tp + n_fp
    precision = n_tp / denom if denom > 0 else 0.0
    # P(A) = A-tier GT 매칭 TP / (A-tier GT 매칭 TP + FP)
    denom_a = n_tp_a + n_fp
    precision_a = n_tp_a / denom_a if denom_a > 0 else 0.0
    f1 = fbeta(precision, recall, 1.0)
    f_beta = fbeta(precision, recall, beta)
    f1_a = fbeta(precision_a, recall_a, 1.0)
    f_beta_a = fbeta(precision_a, recall_a, beta)

    return {
        'window': window,
        'recall': recall, 'precision': precision, 'f1': f1, 'f_beta': f_beta,
        'n_gt': n_gt, 'n_hit': len(hit_gt),
        'n_gt_a': len(gt_a), 'hit_a': hit_a, 'recall_a': recall_a,
        'n_gt_c': len(gt_c), 'hit_c': hit_c, 'recall_c': recall_c,
        'precision_a': precision_a, 'f1_a': f1_a, 'f_beta_a': f_beta_a,
        'n_tp': n_tp, 'n_tp_a': n_tp_a, 'n_fp': n_fp, 'n_excluded': n_excluded,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--window", type=int, default=None, help="단일 window (미지정 시 0/3/5/7 비교)")
    parser.add_argument("--threshold", type=int, default=None, help="mentions_on_date A/C 임계치 (기본: csv 그대로)")
    args = parser.parse_args()

    if args.run:
        run_dir = Path(args.run)
    else:
        runs = sorted(Path("logs/runs").glob("*_Q0.001_R10.0"))
        if not runs:
            print("No Q=0.001 R=10 run found"); return
        run_dir = runs[-1]
        print(f"Using run: {run_dir.name}")

    # Pre-load shared data
    gt = load_gt_all()
    tiers = load_tiers(args.threshold)

    # Load verified once
    dates = sorted([d for d in os.listdir(run_dir / "by_date")
                    if (run_dir / "by_date" / d).is_dir()])
    all_verified = []
    for d in dates:
        vf = run_dir / "by_date" / d / "verified.csv"
        if not vf.exists() or vf.stat().st_size < 10: continue
        df = pd.read_csv(vf)
        if df.empty: continue
        df['run_date'] = d
        all_verified.append(df)
    if not all_verified:
        print("No verified.csv found"); return
    verified = pd.concat(all_verified, ignore_index=True)
    verified['city_l'] = verified['city'].astype(str).str.lower().str.strip()

    windows = [args.window] if args.window is not None else [0, 3, 5, 7]
    results = []
    for w in windows:
        r = evaluate_windowed(run_dir, window=w, gt=gt, verified=verified, tiers=tiers)
        if r:
            results.append(r)

    beta = 1.5
    print(f"\n{'='*70}")
    print(f"Run: {run_dir.name}")
    thr = args.threshold or 10
    print(f"Tier: A = mentions≥{thr} ({results[0]['n_gt_a']}개), C = mentions<{thr} ({results[0]['n_gt_c']}개)")
    print(f"{'='*70}")
    print(f"{'Window':<8} {'R(A)':>7} {'R(C)':>7} {'P(전체)':>8} {'P(A)':>7} {'F1(A)':>7} {'F1.5(A)':>8} {'TP/FP':>8}")
    print(f"{'-'*70}")
    for r in results:
        print(f"  +{r['window']}d   "
              f"{r['recall_a']:>6.1%}  "
              f"{r['recall_c']:>6.1%}  "
              f"{r['precision']:>7.1%}  "
              f"{r['precision_a']:>6.1%}  "
              f"{r['f1_a']:>6.3f}  "
              f"{r['f_beta_a']:>7.3f}  "
              f"{r['n_tp']}/{r['n_fp']}")
    print(f"{'='*70}")
    print(f"  (baseline exact-date Q=0.001 R=10: R=47%, P=64.8%, F1.5=0.508)")


if __name__ == "__main__":
    main()
