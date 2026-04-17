#!/usr/bin/env python3
"""
Level 1 Kalman Spatial Evaluation
─────────────────────────────────
- 신규 GT의 위경도 기준(반경 km)으로 Kalman 성능을 평가한다.
- 이름 exact match 대신 "공간적 일치"를 사용한다.
- 필요 시 Q/R ratio grid search로 더 나은 Kalman 파라미터를 찾는다.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import MIN_HISTORY, OUTPUT_DIR
from pipeline.conflict_index import compute_conflict_index, kalman_innovation
from pipeline.gdelt_fetcher import load_all_data
from pipeline.ground_truth_loader import load_ground_truth


DEFAULT_GT_GLOB = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "ground_truth"
    / "2026*.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kalman spatial GT 평가")
    parser.add_argument(
        "--ground-truth",
        type=str,
        nargs="*",
        default=[str(DEFAULT_GT_GLOB)],
        help="ground truth CSV glob 또는 경로",
    )
    parser.add_argument(
        "--radius-km",
        type=float,
        default=50.0,
        help="GT와 예측을 정답으로 볼 반경(km)",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=MIN_HISTORY,
        help="도시별 최소 시계열 길이",
    )
    parser.add_argument(
        "--q-ratios",
        type=float,
        nargs="*",
        default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        help="Kalman Q ratio 후보들",
    )
    parser.add_argument(
        "--r-ratios",
        type=float,
        nargs="*",
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
        help="Kalman R ratio 후보들",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="*",
        default=[1, 3, 5, 10],
        help="평가할 Top-K 목록",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="저장 파일명",
    )
    return parser.parse_args()


def _normalize_city(value: str) -> str:
    return str(value).strip().lower()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return 2 * radius_km * asin(sqrt(a))


def _detect_kalman(
    city_daily: pd.DataFrame,
    q_ratio: float,
    r_ratio: float,
    min_history: int,
) -> pd.DataFrame:
    results = []
    for city, group in city_daily.groupby("city"):
        group = group.sort_values("date").reset_index(drop=True)
        signal = group["conflict_index"].values.astype(float)
        if len(signal) < min_history:
            continue

        init_window = min(30, len(signal))
        init_var = max(float(np.var(signal[:init_window])), 1.0)
        kf = kalman_innovation(
            signal,
            Q=init_var * q_ratio,
            R=init_var * r_ratio,
        )

        group = group.copy()
        group["innov_z"] = kf["norm_innov"]
        results.append(group)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    combined["date"] = combined["date"].astype(str)
    combined["city_norm"] = combined["city"].astype(str).map(_normalize_city)
    return combined.sort_values(["date", "innov_z"], ascending=[True, False])


def _first_hit_rank(
    day_pred: pd.DataFrame,
    day_truth: pd.DataFrame,
    radius_km: float,
) -> tuple[int | None, dict | None]:
    if day_pred.empty or day_truth.empty:
        return None, None

    for idx, pred in enumerate(day_pred.itertuples(index=False), start=1):
        for truth in day_truth.itertuples(index=False):
            distance_km = _haversine_km(
                float(pred.lat),
                float(pred.lon),
                float(truth.gt_lat),
                float(truth.gt_lon),
            )
            if distance_km <= radius_km:
                return idx, {
                    "pred_city": pred.city,
                    "truth_city": truth.city,
                    "distance_km": round(distance_km, 2),
                }
    return None, None


def _greedy_match_topk(
    day_pred: pd.DataFrame,
    day_truth: pd.DataFrame,
    k: int,
    radius_km: float,
) -> tuple[int, list[dict]]:
    topk = day_pred.head(k).copy()
    remaining_truth = set(day_truth.index.tolist())
    matched_pairs: list[dict] = []

    for rank, pred in enumerate(topk.itertuples(), start=1):
        best_truth_idx = None
        best_distance = None
        for truth_idx in list(remaining_truth):
            truth = day_truth.loc[truth_idx]
            distance_km = _haversine_km(
                float(pred.lat),
                float(pred.lon),
                float(truth["gt_lat"]),
                float(truth["gt_lon"]),
            )
            if distance_km > radius_km:
                continue
            if best_distance is None or distance_km < best_distance:
                best_truth_idx = truth_idx
                best_distance = distance_km

        if best_truth_idx is None:
            continue

        truth = day_truth.loc[best_truth_idx]
        remaining_truth.remove(best_truth_idx)
        matched_pairs.append(
            {
                "rank": rank,
                "pred_city": pred.city,
                "truth_city": truth["city"],
                "distance_km": round(float(best_distance), 2),
            }
        )

    return len(matched_pairs), matched_pairs


def evaluate_spatial_predictions(
    model_results: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    score_col: str,
    radius_km: float,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    if model_results.empty:
        return {
            "dates_evaluated": 0,
            "avg_mrr": 0.0,
            "avg_rank_of_first_hit": None,
            "precision_at_k": {f"p@{k}": 0.0 for k in k_values},
            "recall_at_k": {f"r@{k}": 0.0 for k in k_values},
            "per_date": [],
        }

    gt = ground_truth_df.copy()
    gt["date"] = gt["date"].astype(str)
    gt = gt.dropna(subset=["gt_lat", "gt_lon"]).copy()

    eval_dates = sorted(set(model_results["date"].astype(str)) & set(gt["date"].astype(str)))
    if not eval_dates:
        return {
            "dates_evaluated": 0,
            "avg_mrr": 0.0,
            "avg_rank_of_first_hit": None,
            "precision_at_k": {f"p@{k}": 0.0 for k in k_values},
            "recall_at_k": {f"r@{k}": 0.0 for k in k_values},
            "per_date": [],
        }

    precision_scores = defaultdict(list)
    recall_scores = defaultdict(list)
    reciprocal_ranks = []
    first_ranks = []
    per_date = []
    max_k = max(k_values)

    for date in eval_dates:
        day_truth = gt.loc[gt["date"].astype(str) == date].copy().reset_index(drop=True)
        day_pred = model_results.loc[model_results["date"].astype(str) == date].copy()
        day_pred = day_pred.dropna(subset=["lat", "lon"]).sort_values(score_col, ascending=False)

        first_hit_rank, first_hit_detail = _first_hit_rank(day_pred, day_truth, radius_km)
        if first_hit_rank is None:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / first_hit_rank)
            first_ranks.append(first_hit_rank)

        ranked_city_labels = day_pred["city"].astype(str).tolist()
        day_metrics = {
            "date": date,
            "truth_count": int(len(day_truth)),
            "truth_cities": sorted(day_truth["city"].astype(str).unique().tolist()),
            "prediction_count": int(len(day_pred)),
            "first_hit_rank": first_hit_rank,
            "first_hit_detail": first_hit_detail,
            f"top_{max_k}_predictions": ranked_city_labels[:max_k],
        }

        for k in k_values:
            hits, matched_pairs = _greedy_match_topk(day_pred, day_truth, k, radius_km)
            precision = hits / max(k, 1)
            recall = hits / max(len(day_truth), 1)
            precision_scores[k].append(precision)
            recall_scores[k].append(recall)
            day_metrics[f"top_{k}"] = ranked_city_labels[:k]
            day_metrics[f"hits@{k}"] = hits
            day_metrics[f"matched_truth@{k}"] = [pair["truth_city"] for pair in matched_pairs]
            day_metrics[f"matched_pairs@{k}"] = matched_pairs
            day_metrics[f"p@{k}"] = precision
            day_metrics[f"r@{k}"] = recall

        per_date.append(day_metrics)

    return {
        "dates_evaluated": len(eval_dates),
        "avg_mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "avg_rank_of_first_hit": float(np.mean(first_ranks)) if first_ranks else None,
        "precision_at_k": {
            f"p@{k}": float(np.mean(precision_scores[k])) if precision_scores[k] else 0.0
            for k in k_values
        },
        "recall_at_k": {
            f"r@{k}": float(np.mean(recall_scores[k])) if recall_scores[k] else 0.0
            for k in k_values
        },
        "per_date": per_date,
    }


def compute_spatial_candidate_coverage(
    candidate_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    radius_km: float,
) -> dict:
    gt = ground_truth_df.copy()
    gt["date"] = gt["date"].astype(str)
    gt = gt.dropna(subset=["gt_lat", "gt_lon"]).copy()
    candidate_df = candidate_df.copy()
    candidate_df["date"] = candidate_df["date"].astype(str)
    candidate_df = candidate_df.dropna(subset=["lat", "lon"]).copy()

    rows = []
    reachable_total = 0
    for date, group in gt.groupby("date"):
        day_pred = candidate_df.loc[candidate_df["date"] == date]
        reachable = 0
        unreachable_cities = []
        for truth in group.itertuples(index=False):
            matched = False
            for pred in day_pred.itertuples(index=False):
                distance_km = _haversine_km(
                    float(pred.lat),
                    float(pred.lon),
                    float(truth.gt_lat),
                    float(truth.gt_lon),
                )
                if distance_km <= radius_km:
                    matched = True
                    break
            if matched:
                reachable += 1
            else:
                unreachable_cities.append(str(truth.city))

        reachable_total += reachable
        rows.append(
            {
                "date": str(date),
                "reachable": int(reachable),
                "total": int(len(group)),
                "coverage": float(reachable / max(len(group), 1)),
                "unreachable_cities": sorted(unreachable_cities),
            }
        )

    total = int(len(gt))
    return {
        "reachable_truth_rows": int(reachable_total),
        "total_truth_rows": total,
        "overall_coverage": float(reachable_total / max(total, 1)),
        "per_date": rows,
    }


def _save_payload(payload: dict, output_name: str | None) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"level1_kalman_spatial_eval_{stamp}.json"

    save_path = OUTPUT_DIR / output_name
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return save_path


def main() -> None:
    args = parse_args()
    gt_result = load_ground_truth(args.ground_truth)
    ground_truth_df = gt_result.dataframe.copy()
    if not {"gt_lat", "gt_lon"}.issubset(ground_truth_df.columns):
        raise ValueError("spatial 평가에는 ground truth의 Lat/Long 컬럼이 필요합니다.")

    target_dates = sorted(ground_truth_df["date"].astype(str).unique().tolist())
    raw = load_all_data(target_date=max(target_dates))
    if raw.empty:
        raise RuntimeError("분석 가능한 GDELT 데이터가 없습니다.")

    city_daily = compute_conflict_index(raw)
    if city_daily.empty:
        raise RuntimeError("도시별 갈등 지수를 생성하지 못했습니다.")

    print("\n=== Kalman Spatial Evaluation ===")
    print(f"radius_km      : {args.radius_km}")
    print(f"target dates   : {target_dates}")
    print(f"ground truth   : {len(gt_result.file_summaries)} files")
    if gt_result.date_mismatches:
        print("  [주의] 파일명과 SQLDATE가 다른 ground truth 파일:")
        for item in gt_result.date_mismatches:
            print(
                f"    - {item['file']}: filename={item['file_date']} / SQLDATE={item['sql_date']}"
            )

    coverage = compute_spatial_candidate_coverage(
        city_daily[city_daily["date"].astype(str).isin(target_dates)].copy(),
        ground_truth_df,
        radius_km=args.radius_km,
    )

    rows = []
    for q_ratio in args.q_ratios:
        for r_ratio in args.r_ratios:
            pred = _detect_kalman(
                city_daily,
                q_ratio=q_ratio,
                r_ratio=r_ratio,
                min_history=args.min_history,
            )
            pred = pred[pred["date"].astype(str).isin(target_dates)].copy()
            ev = evaluate_spatial_predictions(
                pred,
                ground_truth_df,
                score_col="innov_z",
                radius_km=args.radius_km,
                k_values=tuple(args.k_values),
            )
            rows.append(
                {
                    "q_ratio": float(q_ratio),
                    "r_ratio": float(r_ratio),
                    "avg_mrr": ev["avg_mrr"],
                    "avg_first_hit_rank": ev["avg_rank_of_first_hit"],
                    **ev["precision_at_k"],
                    **ev["recall_at_k"],
                }
            )

    grid_df = pd.DataFrame(rows).sort_values(
        ["avg_mrr", f"r@{max(args.k_values)}", f"p@{max(args.k_values)}"],
        ascending=False,
    ).reset_index(drop=True)

    best = grid_df.iloc[0].to_dict()
    best_pred = _detect_kalman(
        city_daily,
        q_ratio=float(best["q_ratio"]),
        r_ratio=float(best["r_ratio"]),
        min_history=args.min_history,
    )
    best_pred = best_pred[best_pred["date"].astype(str).isin(target_dates)].copy()
    best_eval = evaluate_spatial_predictions(
        best_pred,
        ground_truth_df,
        score_col="innov_z",
        radius_km=args.radius_km,
        k_values=tuple(args.k_values),
    )

    print("\n[Coverage]")
    print(
        f"  reachable / total : {coverage['reachable_truth_rows']} / "
        f"{coverage['total_truth_rows']} ({coverage['overall_coverage']:.1%})"
    )
    for row in coverage["per_date"]:
        print(
            f"  {row['date']} | reachable={row['reachable']}/{row['total']} "
            f"({row['coverage']:.1%})"
        )

    print("\n[Best Kalman by Spatial GT]")
    print(
        f"  q_ratio={best['q_ratio']:.4f} | r_ratio={best['r_ratio']:.4f} | "
        f"avg_mrr={best['avg_mrr']:.4f} | avg_first_hit_rank={best['avg_first_hit_rank']}"
    )
    for key, value in best_eval["precision_at_k"].items():
        print(f"  {key:12s}: {value:.4f}")
    for key, value in best_eval["recall_at_k"].items():
        print(f"  {key:12s}: {value:.4f}")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "radius_km": args.radius_km,
        "ground_truth_paths": args.ground_truth,
        "ground_truth_date_mismatches": gt_result.date_mismatches,
        "target_dates": target_dates,
        "coverage": coverage,
        "grid_search": grid_df.to_dict("records"),
        "best_params": best,
        "best_evaluation": best_eval,
    }
    save_path = _save_payload(payload, args.output_name)
    print(f"\n[저장 완료] {save_path}")


if __name__ == "__main__":
    main()
