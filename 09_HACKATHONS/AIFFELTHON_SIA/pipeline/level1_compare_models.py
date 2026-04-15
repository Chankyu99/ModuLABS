#!/usr/bin/env python3
"""
Level 1 — Kalman vs ARIMA 비교 실행 스크립트
──────────────────────────────────────────────
- 동일한 conflict_index 시계열에 대해 Kalman / ARIMA 성능을 비교
- ground_truth_combined_0327_0401.csv를 기본 평가셋으로 사용
- 결과를 콘솔 요약 + JSON 파일로 저장
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from pipeline.config import OUTPUT_DIR
from pipeline.conflict_index import compute_conflict_index
from pipeline.gdelt_fetcher import load_all_data
from pipeline.ground_truth_loader import load_ground_truth
from pipeline.level1_arima import compare_kalman_vs_arima


DEFAULT_GT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "ground_truth"
    / "ground_truth_combined_0327_0401.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Level 1 Kalman vs ARIMA 비교")
    parser.add_argument(
        "--ground-truth",
        type=str,
        nargs="*",
        default=[str(DEFAULT_GT_PATH)] if DEFAULT_GT_PATH.exists() else [],
        help="ground truth CSV 경로",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=30,
        help="도시별 최소 시계열 길이",
    )
    parser.add_argument(
        "--transform",
        choices=["log1p", "none"],
        default="log1p",
        help="ARIMA 입력 전처리 방식",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="*",
        default=[1, 3, 5, 10],
        help="평가할 k 값 목록 (예: --k-values 1 3 5)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="저장 파일명 (기본값: 자동 생성)",
    )
    return parser.parse_args()


def _print_metric_block(title: str, evaluation: dict) -> None:
    print(f"\n[{title}]")
    print(f"  dates_evaluated       : {evaluation.get('dates_evaluated', 0)}")
    print(f"  avg_mrr               : {evaluation.get('avg_mrr', 0.0):.4f}")
    avg_rank = evaluation.get("avg_rank_of_first_hit")
    if avg_rank is None:
        print("  avg_first_hit_rank    : N/A")
    else:
        print(f"  avg_first_hit_rank    : {avg_rank:.2f}")

    precision = evaluation.get("precision_at_k", {})
    recall = evaluation.get("recall_at_k", {})
    for key, value in precision.items():
        print(f"  {key:22s}: {value:.4f}")
    for key, value in recall.items():
        print(f"  {key:22s}: {value:.4f}")


def _save_result(payload: dict, output_name: str | None = None) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"level1_model_compare_{stamp}.json"

    save_path = OUTPUT_DIR / output_name
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return save_path


def main() -> None:
    args = parse_args()

    gt_result = load_ground_truth(args.ground_truth)
    ground_truth_df = gt_result.dataframe.copy()
    target_dates = sorted(ground_truth_df["date"].unique().tolist())
    max_target_date = max(target_dates)

    print("\n=== Level 1 모델 비교 실행 ===")
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
    print(f"target dates : {target_dates[0]} ~ {target_dates[-1]} ({len(target_dates)}일)")
    print(f"min_history  : {args.min_history}")
    print(f"transform    : {args.transform}")
    print(f"k_values     : {args.k_values}")
    print(f"positive-only: {gt_result.is_positive_only}")

    raw = load_all_data(target_date=max_target_date)
    if raw.empty:
        raise RuntimeError("분석 가능한 GDELT 데이터가 없습니다.")

    city_daily = compute_conflict_index(raw)
    if city_daily.empty:
        raise RuntimeError("도시별 일 단위 갈등 지수를 생성하지 못했습니다.")

    comparison = compare_kalman_vs_arima(
        city_daily=city_daily,
        ground_truth_df=ground_truth_df,
        target_dates=target_dates,
        min_history=args.min_history,
        transform=args.transform,
        k_values=tuple(args.k_values),
        allow_fallback_all_cities=gt_result.is_positive_only,
    )

    payload = {
        "generated_at": datetime.now().isoformat(),
        "ground_truth_paths": [str(path) for path in args.ground_truth],
        "ground_truth_positive_only": gt_result.is_positive_only,
        "ground_truth_date_mismatches": gt_result.date_mismatches,
        "target_dates": target_dates,
        "city_daily_rows": int(len(city_daily)),
        "comparison": comparison,
    }

    _print_metric_block("Kalman", comparison["kalman"]["evaluation"])
    _print_metric_block("ARIMA", comparison["arima"]["evaluation"])

    save_path = _save_result(payload, output_name=args.output_name)
    print(f"\n[저장 완료] {save_path}")


if __name__ == "__main__":
    main()
