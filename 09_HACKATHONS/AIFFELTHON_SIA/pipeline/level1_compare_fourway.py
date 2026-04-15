#!/usr/bin/env python3
"""
Level 1 — 4개 조합 비교 실행 스크립트
─────────────────────────────────────
- A안: 기존 휴리스틱 conflict_index
- B안: ARIMA 친화형 EDA 기반 conflict_index
- K안: Kalman
- R안: ARIMA

총 4개 조합:
- A + K
- A + R
- B + K
- B + R
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import MIN_HISTORY, OUTPUT_DIR, get_risk_level
from pipeline.conflict_index import compute_conflict_index, detect_anomalies as detect_anomalies_kalman
from pipeline.gdelt_fetcher import load_all_data
from pipeline.ground_truth_loader import load_ground_truth
from pipeline.level1_arima import detect_anomalies_arima, evaluate_model_predictions
from pipeline.level1_features import (
    ArimaFriendlyWeights,
    build_city_day_features,
    compute_arima_friendly_conflict_index,
    detect_tasking_with_arima,
)


DEFAULT_GT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "ground_truth"
    / "ground_truth_combined_0327_0401.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Level 1 4개 조합 비교")
    parser.add_argument(
        "--ground-truth",
        type=str,
        nargs="*",
        default=[str(DEFAULT_GT_PATH)] if DEFAULT_GT_PATH.exists() else [],
    )
    parser.add_argument("--min-history", type=int, default=MIN_HISTORY)
    parser.add_argument("--transform", choices=["log1p", "none"], default="log1p")
    parser.add_argument("--k-values", type=int, nargs="*", default=[1, 3, 5, 10])
    parser.add_argument("--target-dates", type=str, nargs="*", default=None)
    parser.add_argument("--max-dates", type=int, default=None, help="앞에서부터 비교할 날짜 수 제한")
    parser.add_argument(
        "--restrict-to-gt-cities",
        action="store_true",
        help="스모크 테스트용: ground truth에 등장한 도시만 대상으로 모델링",
    )
    parser.add_argument("--output-name", type=str, default=None)
    return parser.parse_args()


def _safe_iqr(values: pd.Series) -> float:
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = float(q3 - q1)
    return iqr if iqr > 1e-8 else 1.0


def _add_level_z(df: pd.DataFrame, base_index_col: str) -> pd.DataFrame:
    result = df.copy()

    def _transform(group: pd.DataFrame) -> pd.DataFrame:
        median = float(group[base_index_col].median())
        iqr = _safe_iqr(group[base_index_col])
        group["level_z"] = (group[base_index_col] - median) / iqr
        return group

    return result.groupby("city", group_keys=False).apply(_transform).reset_index(drop=True)


def _add_tasking_score(
    df: pd.DataFrame,
    anomaly_col: str,
    base_index_col: str,
    score_name: str = "tasking_score",
    surprise_weight: float = 0.70,
    level_weight: float = 0.30,
) -> pd.DataFrame:
    result = _add_level_z(df, base_index_col=base_index_col)
    result["surprise_score"] = np.maximum(0.0, result[anomaly_col].fillna(0.0))
    result["level_score"] = np.maximum(0.0, result["level_z"].fillna(0.0))
    result[score_name] = (
        surprise_weight * result["surprise_score"]
        + level_weight * result["level_score"]
    )

    risk_info = result[score_name].apply(get_risk_level)
    result["risk_level"] = risk_info.apply(lambda x: x["level"])
    result["risk_label"] = risk_info.apply(lambda x: x["label"])
    result["risk_emoji"] = risk_info.apply(lambda x: x["emoji"])
    result["risk_guide"] = risk_info.apply(lambda x: x["guide"])
    result["is_anomaly"] = result["risk_level"] >= 1
    return result


def _prepare_target_dates(gt: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    if args.target_dates:
        dates = [str(d) for d in args.target_dates]
    else:
        dates = sorted(gt["date"].astype(str).unique().tolist())

    if args.max_dates is not None:
        dates = dates[: args.max_dates]
    return dates


def _print_metric_block(title: str, evaluation: dict) -> None:
    print(f"\n[{title}]")
    print(f"  dates_evaluated       : {evaluation.get('dates_evaluated', 0)}")
    print(f"  avg_mrr               : {evaluation.get('avg_mrr', 0.0):.4f}")
    avg_rank = evaluation.get("avg_rank_of_first_hit")
    print(
        f"  avg_first_hit_rank    : {'N/A' if avg_rank is None else f'{avg_rank:.2f}'}"
    )
    for key, value in evaluation.get("precision_at_k", {}).items():
        print(f"  {key:22s}: {value:.4f}")
    for key, value in evaluation.get("recall_at_k", {}).items():
        print(f"  {key:22s}: {value:.4f}")


def _save_result(payload: dict, output_name: str | None = None) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"level1_fourway_compare_{stamp}.json"

    save_path = OUTPUT_DIR / output_name
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return save_path


def main() -> None:
    args = parse_args()
    gt_result = load_ground_truth(args.ground_truth)
    ground_truth_df = gt_result.dataframe.copy()
    target_dates = _prepare_target_dates(ground_truth_df, args)
    if not target_dates:
        raise RuntimeError("비교할 target_dates가 비어 있습니다.")

    ground_truth_df = ground_truth_df[ground_truth_df["date"].astype(str).isin(target_dates)].copy()
    max_target_date = max(target_dates)

    print("\n=== Level 1 4개 조합 비교 실행 ===")
    print(f"ground truth files : {len(gt_result.file_summaries)}개")
    for item in gt_result.file_summaries:
        print(
            f"  - {item['file']} | schema={item['schema']} | "
            f"dates={item['dates']} | cities={item['cities']}"
        )
    if gt_result.date_mismatches:
        print("  [주의] 파일명과 SQLDATE가 다른 ground truth 파일:")
        for item in gt_result.date_mismatches:
            print(
                f"    - {item['file']}: filename={item['file_date']} / SQLDATE={item['sql_date']}"
            )
    print(f"target dates : {target_dates}")
    print(f"min_history  : {args.min_history}")
    print(f"transform    : {args.transform}")
    print(f"k_values     : {args.k_values}")
    print(f"gt city only : {args.restrict_to_gt_cities}")
    print(f"positive-only: {gt_result.is_positive_only}")

    raw = load_all_data(target_date=max_target_date)
    if raw.empty:
        raise RuntimeError("분석 가능한 GDELT 데이터가 없습니다.")

    # A안: 기존 휴리스틱 지수
    city_daily_a = compute_conflict_index(raw)
    if args.restrict_to_gt_cities and not city_daily_a.empty:
        gt_cities = set(ground_truth_df["city"].astype(str))
        city_daily_a = city_daily_a[city_daily_a["city"].astype(str).isin(gt_cities)].copy()

    a_k_raw = detect_anomalies_kalman(city_daily_a)
    a_r_raw = detect_anomalies_arima(
        city_daily_a,
        min_history=args.min_history,
        transform=args.transform,
    )

    a_k = _add_tasking_score(a_k_raw, anomaly_col="innov_z", base_index_col="conflict_index")
    a_r = _add_tasking_score(a_r_raw, anomaly_col="arima_z", base_index_col="conflict_index")

    # B안: ARIMA 친화형 EDA 지수
    city_day_b = build_city_day_features(raw)
    if args.restrict_to_gt_cities and not city_day_b.empty:
        gt_cities = set(ground_truth_df["city"].astype(str))
        city_day_b = city_day_b[city_day_b["city"].astype(str).isin(gt_cities)].copy()
    city_index_b = compute_arima_friendly_conflict_index(city_day_b, weights=ArimaFriendlyWeights())

    b_k_input = city_index_b.copy()
    b_k_input["conflict_index"] = b_k_input["conflict_index_arima"]
    b_k_raw = detect_anomalies_kalman(b_k_input)
    b_k = _add_tasking_score(b_k_raw, anomaly_col="innov_z", base_index_col="conflict_index")

    b_r = detect_tasking_with_arima(
        city_index_b,
        min_history=args.min_history,
        transform=args.transform,
        weights=ArimaFriendlyWeights(),
    )

    # 날짜 필터
    date_mask = lambda df: df[df["date"].astype(str).isin(target_dates)].copy()
    a_k = date_mask(a_k)
    a_r = date_mask(a_r)
    b_k = date_mask(b_k)
    b_r = date_mask(b_r)

    k_values = tuple(args.k_values)
    evaluations = {
        "A_Kalman": evaluate_model_predictions(
            a_k,
            "tasking_score",
            ground_truth_df,
            k_values=k_values,
            allow_fallback_all_cities=gt_result.is_positive_only,
        ),
        "A_ARIMA": evaluate_model_predictions(
            a_r,
            "tasking_score",
            ground_truth_df,
            k_values=k_values,
            allow_fallback_all_cities=gt_result.is_positive_only,
        ),
        "B_Kalman": evaluate_model_predictions(
            b_k,
            "tasking_score",
            ground_truth_df,
            k_values=k_values,
            allow_fallback_all_cities=gt_result.is_positive_only,
        ),
        "B_ARIMA": evaluate_model_predictions(
            b_r,
            "tasking_score",
            ground_truth_df,
            k_values=k_values,
            allow_fallback_all_cities=gt_result.is_positive_only,
        ),
    }

    payload = {
        "generated_at": datetime.now().isoformat(),
        "ground_truth_paths": [str(path) for path in args.ground_truth],
        "ground_truth_positive_only": gt_result.is_positive_only,
        "ground_truth_date_mismatches": gt_result.date_mismatches,
        "target_dates": target_dates,
        "config": {
            "min_history": args.min_history,
            "transform": args.transform,
            "k_values": args.k_values,
        },
        "dataset": {
            "raw_rows": int(len(raw)),
            "heuristic_city_daily_rows": int(len(city_daily_a)),
            "heuristic_cities": int(city_daily_a["city"].nunique()) if not city_daily_a.empty else 0,
            "eda_city_day_rows": int(len(city_day_b)),
            "eda_city_index_rows": int(len(city_index_b)),
            "eda_cities": int(city_index_b["city"].nunique()) if not city_index_b.empty else 0,
        },
        "evaluations": evaluations,
    }

    for name, evaluation in evaluations.items():
        _print_metric_block(name, evaluation)

    save_path = _save_result(payload, output_name=args.output_name)
    print(f"\n[저장 완료] {save_path}")


if __name__ == "__main__":
    main()
