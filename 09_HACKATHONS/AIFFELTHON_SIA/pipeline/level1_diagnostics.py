#!/usr/bin/env python3
"""
Level 1 진단 스크립트
────────────────────
- 현재 라벨 데이터 기준으로 TP/FP 분포를 요약
- FP error_type 비중을 확인
- 현재 Level 1 필터에서 누락되는 labeled city-day를 식별
- TP/FP feature median 차이를 출력

목적:
- "칼만 파라미터" 문제인지,
- "전처리/필터/도시명 정규화" 문제인지,
- "가중치 설계" 문제인지를 먼저 분리해서 본다.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.conflict_index import compute_conflict_index
from pipeline.config import ACTION_GEO_ALLOWED_COUNTRIES, CONFIRMED_CODES, MONITORED_COUNTRIES, OUTPUT_DIR
from pipeline.gdelt_fetcher import load_all_data
from pipeline.level1_features import (
    build_city_day_features,
    compute_arima_friendly_conflict_index,
)


DEFAULT_GT_PATHS = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "ground_truth"
    / "city_level_ground_truth_20260326_20260330.csv",
    Path(__file__).resolve().parent.parent
    / "data"
    / "ground_truth"
    / "ground_truth_combined_0327_0401.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Level 1 진단 리포트 생성")
    parser.add_argument(
        "--ground-truth",
        type=str,
        nargs="*",
        default=[str(path) for path in DEFAULT_GT_PATHS if path.exists()],
        help="ground truth CSV 경로들",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="JSON 저장 파일명",
    )
    return parser.parse_args()


def _city_key(value: str) -> str:
    return str(value).strip().lower()


def _load_labels(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["date"] = df["date"].astype(str)
        df["city"] = df["city"].astype(str)

        selection = (
            df["selection_decision"].fillna("").astype(str).str.lower()
            if "selection_decision" in df.columns
            else pd.Series("", index=df.index)
        )
        manual = (
            df["manual_label"].fillna("").astype(str).str.upper()
            if "manual_label" in df.columns
            else pd.Series("", index=df.index)
        )

        df["label"] = np.where(
            (selection == "should_keep") | (manual == "TP"),
            1,
            np.where(
                (selection == "should_drop") | (manual == "FP"),
                0,
                np.nan,
            ),
        )
        df["source_file"] = path.name
        frames.append(df)

    labels = pd.concat(frames, ignore_index=True)
    labels = labels[labels["label"].notna()].copy()
    labels["label"] = labels["label"].astype(int)
    labels["city_key"] = labels["city"].map(_city_key)

    # 동일 date/city가 여러 파일에 있으면 positive를 우선한다.
    labels = labels.sort_values(
        ["date", "city_key", "label"],
        ascending=[True, True, False],
    ).drop_duplicates(["date", "city_key"], keep="first")
    return labels.reset_index(drop=True)


def _current_pipeline_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["Actor1CountryCode"].isin(MONITORED_COUNTRIES) | df["Actor2CountryCode"].isin(MONITORED_COUNTRIES))
        & pd.to_numeric(df["EventCode"], errors="coerce").isin(CONFIRMED_CODES)
        & (pd.to_numeric(df["ActionGeo_Type"], errors="coerce") == 4)
        & (df["ActionGeo_CountryCode"].isin(ACTION_GEO_ALLOWED_COUNTRIES))
        & (pd.to_numeric(df["NumSources"], errors="coerce") >= 1)
    )


def _build_filtered_event_aggregates(raw: pd.DataFrame) -> pd.DataFrame:
    filtered = raw.loc[_current_pipeline_mask(raw)].copy()
    filtered["date"] = filtered["SQLDATE"].astype(str).str[:8]
    filtered["city_key"] = (
        filtered["ActionGeo_FullName"]
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
        .map(_city_key)
    )
    filtered["EventRootCode_num"] = pd.to_numeric(filtered["EventRootCode"], errors="coerce").fillna(0).astype(int)
    filtered["AvgTone_num"] = pd.to_numeric(filtered["AvgTone"], errors="coerce").fillna(0.0)
    filtered["Goldstein_num"] = pd.to_numeric(filtered["GoldsteinScale"], errors="coerce").fillna(0.0)
    filtered["negative_tone"] = np.maximum(0.0, -filtered["AvgTone_num"])
    filtered["negative_goldstein"] = np.maximum(0.0, -filtered["Goldstein_num"])

    return (
        filtered.groupby(["date", "city_key"])
        .agg(
            agg_events=("EventCode", "count"),
            agg_mentions_sum=("NumMentions", "sum"),
            agg_sources_sum=("NumSources", "sum"),
            agg_articles_sum=("NumArticles", "sum"),
            agg_negative_tone_mean=("negative_tone", "mean"),
            agg_negative_goldstein_mean=("negative_goldstein", "mean"),
            agg_root18_ratio=("EventRootCode_num", lambda s: float(np.mean(s == 18))),
            agg_root19_ratio=("EventRootCode_num", lambda s: float(np.mean(s == 19))),
        )
        .reset_index()
    )


def _summarize_feature_medians(merged: pd.DataFrame) -> dict:
    feature_cols = [
        "agg_events",
        "agg_mentions_sum",
        "agg_sources_sum",
        "agg_articles_sum",
        "agg_negative_tone_mean",
        "agg_negative_goldstein_mean",
        "agg_root18_ratio",
        "agg_root19_ratio",
        "mi_raw",
        "es_raw",
        "ht_raw",
        "conflict_index_arima",
        "heur_conflict_index",
    ]

    available = [col for col in feature_cols if col in merged.columns]
    out: dict[str, dict[str, float | None]] = {}
    for col in available:
        neg = merged.loc[merged["label"] == 0, col].median()
        pos = merged.loc[merged["label"] == 1, col].median()
        out[col] = {
            "negative_median": None if pd.isna(neg) else float(neg),
            "positive_median": None if pd.isna(pos) else float(pos),
            "delta_pos_minus_neg": None if pd.isna(neg) or pd.isna(pos) else float(pos - neg),
        }
    return out


def _diagnose_unmatched_labels(labels: pd.DataFrame, raw: pd.DataFrame, matched_mask: pd.Series) -> list[dict]:
    raw = raw.copy()
    raw["date"] = raw["SQLDATE"].astype(str).str[:8]
    raw["city_key"] = raw["ActionGeo_FullName"].astype(str).str.split(",").str[0].str.strip().map(_city_key)

    unmatched = labels.loc[~matched_mask].copy()
    diagnostics: list[dict] = []

    for _, row in unmatched.iterrows():
        day = raw[raw["date"] == row["date"]].copy()
        city_rows = day[day["city_key"] == row["city_key"]].copy()

        reason = "city_not_found_in_raw"
        detail = {}

        if not city_rows.empty:
            country_pass = (
                city_rows["Actor1CountryCode"].isin(MONITORED_COUNTRIES)
                | city_rows["Actor2CountryCode"].isin(MONITORED_COUNTRIES)
            )
            code_pass = pd.to_numeric(city_rows["EventCode"], errors="coerce").isin(CONFIRMED_CODES)
            geo_pass = pd.to_numeric(city_rows["ActionGeo_Type"], errors="coerce") == 4
            sources_pass = pd.to_numeric(city_rows["NumSources"], errors="coerce") >= 2

            detail = {
                "raw_rows": int(len(city_rows)),
                "country_pass_rows": int(country_pass.sum()),
                "confirmed_code_rows": int(code_pass.sum()),
                "geo_type4_rows": int(geo_pass.sum()),
                "sources_ge_2_rows": int(sources_pass.sum()),
                "root_codes": sorted(
                    pd.to_numeric(city_rows["EventRootCode"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                ),
                "event_codes": sorted(
                    pd.to_numeric(city_rows["EventCode"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                )[:20],
            }

            if code_pass.any() and not sources_pass.any():
                reason = "blocked_by_min_sources"
            elif not code_pass.any():
                reason = "blocked_by_confirmed_codes"
            elif not country_pass.any():
                reason = "blocked_by_monitored_countries"
            elif not geo_pass.any():
                reason = "blocked_by_actiongeo_type"
            else:
                reason = "blocked_by_combined_filters"

        diagnostics.append(
            {
                "date": str(row["date"]),
                "city": str(row["city"]),
                "label": int(row["label"]),
                "source_file": str(row["source_file"]),
                "diagnosis": reason,
                "detail": detail,
            }
        )

    return diagnostics


def _print_summary(payload: dict) -> None:
    labels = payload["labels"]
    print("\n=== Level 1 Diagnostics ===")
    print(f"labeled rows        : {labels['count']}")
    print(f"positives / negatives : {labels['positives']} / {labels['negatives']}")

    print("\n[FP error_type]")
    for key, value in payload["fp_error_types"].items():
        print(f"  {key:24s}: {value}")

    unmatched = payload["unmatched"]
    print(f"\n[unmatched labeled city-days] {unmatched['count']}")
    for item in unmatched["rows"][:12]:
        print(
            f"  {item['date']} | {item['city']:<15s} | label={item['label']} | "
            f"{item['diagnosis']}"
        )

    print("\n[feature medians: positive - negative delta]")
    for key, item in payload["feature_medians"].items():
        delta = item["delta_pos_minus_neg"]
        delta_str = "N/A" if delta is None else f"{delta:+.3f}"
        print(f"  {key:28s}: {delta_str}")


def _save_payload(payload: dict, output_name: str | None) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"level1_diagnostics_{stamp}.json"

    save_path = OUTPUT_DIR / output_name
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return save_path


def main() -> None:
    args = parse_args()
    gt_paths = [Path(path) for path in args.ground_truth]
    if not gt_paths:
        raise RuntimeError("ground truth 파일이 없습니다.")
    for path in gt_paths:
        if not path.exists():
            raise FileNotFoundError(f"ground truth 파일이 없습니다: {path}")

    labels = _load_labels(gt_paths)
    raw = load_all_data(target_date=str(labels["date"].max()))
    if raw.empty:
        raise RuntimeError("분석 가능한 GDELT 데이터가 없습니다.")

    agg = _build_filtered_event_aggregates(raw)

    city_day = build_city_day_features(raw)
    city_day["city_key"] = city_day["city"].map(_city_key)
    city_index = compute_arima_friendly_conflict_index(city_day)
    city_index["city_key"] = city_index["city"].map(_city_key)

    heuristic = compute_conflict_index(raw)
    heuristic["city_key"] = heuristic["city"].astype(str).str.split(",").str[0].str.strip().map(_city_key)
    heuristic = heuristic.rename(columns={"conflict_index": "heur_conflict_index"})

    merged = labels.merge(agg, on=["date", "city_key"], how="left")
    merged = merged.merge(
        city_index[["date", "city_key", "mi_raw", "es_raw", "ht_raw", "conflict_index_arima"]],
        on=["date", "city_key"],
        how="left",
    )
    merged = merged.merge(
        heuristic[["date", "city_key", "heur_conflict_index"]],
        on=["date", "city_key"],
        how="left",
    )

    matched_mask = merged["agg_events"].notna()
    unmatched_rows = _diagnose_unmatched_labels(labels, raw, matched_mask)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "ground_truth_paths": [str(path) for path in gt_paths],
        "labels": {
            "count": int(len(labels)),
            "positives": int(labels["label"].sum()),
            "negatives": int((1 - labels["label"]).sum()),
        },
        "fp_error_types": {
            str(key): int(value)
            for key, value in labels.loc[labels["label"] == 0, "error_type"].fillna("NA").value_counts().items()
        },
        "matched": {
            "count": int(matched_mask.sum()),
            "total": int(len(merged)),
        },
        "unmatched": {
            "count": int((~matched_mask).sum()),
            "rows": unmatched_rows,
        },
        "feature_medians": _summarize_feature_medians(merged.loc[matched_mask].copy()),
    }

    _print_summary(payload)
    save_path = _save_payload(payload, args.output_name)
    print(f"\n[저장 완료] {save_path}")


if __name__ == "__main__":
    main()
