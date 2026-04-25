"""
Q/R grid search runner.

Evaluates each (Q, R) combo over all GT-available dates, logs per-date metrics
via RunLogger, and writes a tuning summary across all combos.

Usage:
  python run_experiment.py                        # all GT dates, default grid
  python run_experiment.py --dates 20260228 20260302
  python run_experiment.py --grid small           # smaller grid for quick check
"""

import argparse
import json
import pandas as pd

from src.config import MAIN_PATH, URL_PATH, PROJECT_ROOT
from src.kalman_filter import compute_conflict_index, detect_anomalies
from src.llm_verification import verify_anomalies_with_llm
from src.evaluator import ModelEvaluator
from src.run_logger import RunLogger

GT_DIR = PROJECT_ROOT / "data" / "ground_truth"

GRIDS = {
    "default": [(q, r) for q in [0.01, 0.05, 0.1] for r in [0.1, 0.5, 1.0]],
    "small":   [(0.01, 0.5), (0.1, 1.0)],
    "wide":    [(q, r) for q in [0.001, 0.01, 0.1, 1.0] for r in [0.01, 0.1, 1.0, 10.0]],
    "fine":    [(q, r) for q in [0.0001, 0.0005, 0.001] for r in [5.0, 10.0, 20.0, 50.0]],
    # auto R (per-city init_var * KALMAN_R_RATIO) — restores small-city sensitivity
    "auto_r":  [(0.0001, -1.0), (0.0005, -1.0), (0.001, -1.0), (0.01, -1.0), (-1.0, -1.0)],
    # aggressive: large Q + small R — fast response, trust observations
    "aggressive": [(q, r) for q in [0.1, 0.5, 1.0] for r in [0.1, 0.5, 1.0]],
    # mean/max aggregation: global std ~0.22 so R scaled down ~50x vs sum
    "agg_small": [(0.001, 0.05), (0.001, 0.1), (0.01, 0.1), (0.01, 0.5)],
    "best":      [(0.001, 10.0)],
}


def _log_threshold_counts(anomalies_for_date: pd.DataFrame, target_date: str) -> None:
    """Print per-risk_level anomaly counts for a single date."""
    if anomalies_for_date.empty:
        print(f"    [{target_date}] anomalies total: 0")
        return
    counts = anomalies_for_date.groupby('risk_level').size()
    print(
        f"    [{target_date}] anomalies "
        f"🔴{counts.get(3, 0)} 🟠{counts.get(2, 0)} 🟡{counts.get(1, 0)} "
        f"= {int(counts.sum())} total"
    )


def run_one_combo(q: float, r: float, raw_df: pd.DataFrame, url_df: pd.DataFrame,
                  evaluator: ModelEvaluator, eval_dates: list, top_k: int) -> dict:
    print(f"\n{'='*60}\n>>> Q={q} R={r}\n{'='*60}")

    city_daily, raw_filtered = compute_conflict_index(raw_df, is_train=False, manual_q=q, manual_r=r)
    if city_daily.empty:
        print("  [WARN] no city_daily output — skipping")
        return {"q": q, "r": r, "skipped": True}

    logger = RunLogger(q=q, r=r, extra_config={"top_k": top_k, "eval_dates": eval_dates})

    for target_date in eval_dates:
        anomalies_today = detect_anomalies(city_daily, target_date)
        _log_threshold_counts(anomalies_today, target_date)

        if anomalies_today.empty:
            metrics = evaluator.evaluate(anomalies_today.assign(llm_status='UNVERIFIED'), target_date)
            logger.log_date(target_date, metrics, anomalies_today)
            continue

        verified = verify_anomalies_with_llm(
            anomalies_today, raw_filtered, url_df, target_date, top_k=top_k
        )
        metrics = evaluator.evaluate(verified, target_date)
        logger.log_date(target_date, metrics, anomalies_today)

    return logger.finalize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", nargs="*", default=None, help="Subset of YYYYMMDD to evaluate (default: all GT dates).")
    parser.add_argument("--grid", default="default", choices=list(GRIDS), help="Q/R grid preset.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top anomalies sent to LLM.")
    args = parser.parse_args()

    print("[SYSTEM] Loading datasets...")
    raw_df = pd.read_parquet(MAIN_PATH)
    url_df = pd.read_parquet(URL_PATH)

    # Build name → FeatureIDs map from raw GDELT for alias fallback (e.g., Esfahan↔Isfahan)
    name_col = 'ActionGeo_FullName'
    fid_col  = 'ActionGeo_FeatureID'
    name_to_fids = {}
    if name_col in raw_df.columns and fid_col in raw_df.columns:
        tmp = raw_df[[name_col, fid_col]].dropna()
        tmp[name_col] = tmp[name_col].astype(str).str.strip().str.lower()
        def _to_int_set(s):
            out = set()
            for x in s.unique():
                try:
                    out.add(int(x))
                except (TypeError, ValueError):
                    continue
            return out
        name_to_fids = tmp.groupby(name_col)[fid_col].apply(_to_int_set).to_dict()
        print(f"[SYSTEM] Built name→FeatureID alias map ({len(name_to_fids)} names)")

    evaluator = ModelEvaluator(GT_DIR, name_to_fids=name_to_fids)

    eval_dates = args.dates or evaluator.available_dates()
    print(f"[SYSTEM] Evaluating {len(eval_dates)} date(s): {eval_dates}")

    grid = GRIDS[args.grid]
    print(f"[SYSTEM] Grid '{args.grid}' — {len(grid)} (Q,R) combos")

    tuning_results = []
    for q, r in grid:
        summary = run_one_combo(q, r, raw_df, url_df, evaluator, eval_dates, args.top_k)
        tuning_results.append(summary)

    # Aggregate summary across grid
    out_path = PROJECT_ROOT / "logs" / "tuning_summary.json"
    out_path.write_text(json.dumps(tuning_results, ensure_ascii=False, indent=2, default=float))
    print(f"\n[DONE] tuning summary → {out_path.relative_to(PROJECT_ROOT)}")

    # Print leaderboard sorted by avg_f_beta
    ranked = sorted([t for t in tuning_results if not t.get("skipped")],
                    key=lambda x: x.get("avg_f_beta", 0), reverse=True)
    print("\n[LEADERBOARD — avg F_beta]")
    for i, t in enumerate(ranked[:10], 1):
        print(f"  {i}. Q={t['q']} R={t['r']}  avg F={t['avg_f_beta']:.3f}  "
              f"P={t['avg_precision']:.2%} R={t['avg_recall']:.2%}  minF={t['min_f_beta']:.3f}")


if __name__ == "__main__":
    main()
