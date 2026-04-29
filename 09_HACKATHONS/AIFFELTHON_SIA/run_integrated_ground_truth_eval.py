#!/usr/bin/env python3
"""
Integrated pipeline + Level2a Ground Truth evaluator.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.archive.city_utils import normalize_city_key
from pipeline.integrated_pipeline import resolve_main_data_path, run_integrated_pipeline


GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "ground_truth"
OUTPUT_DIR = PROJECT_ROOT / "output"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def load_ground_truth_rows(date: str) -> list[dict[str, Any]]:
    path = GROUND_TRUTH_DIR / f"{date}.csv"
    rows: list[dict[str, Any]] = []

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            city = str(row.get("ActionGeo_FullName", "")).strip()
            lat_raw = row.get("Lat")
            lon_raw = row.get("Long")
            if not city or lat_raw in (None, "") or lon_raw in (None, ""):
                continue
            rows.append(
                {
                    "date": date,
                    "city": city,
                    "normalized_city": normalize_city_key(city),
                    "lat": float(lat_raw),
                    "lon": float(lon_raw),
                    "source": str(row.get("Source", "")).strip(),
                    "event_description": str(row.get("Event_Description", "")).strip(),
                }
            )

    return rows


def load_all_ground_truth_rows(dates: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for date in dates:
        rows.extend(load_ground_truth_rows(date))
    return rows


def load_mentions_frame() -> pd.DataFrame:
    main_data_path = resolve_main_data_path()
    raw_df = pd.read_parquet(main_data_path)
    mentions_df = raw_df[
        [
            "SQLDATE",
            "ActionGeo_FullName",
            "ActionGeo_Lat",
            "ActionGeo_Long",
            "NumMentions",
        ]
    ].copy()
    mentions_df["date"] = mentions_df["SQLDATE"].astype(str).str[:8]
    mentions_df["normalized_city"] = mentions_df["ActionGeo_FullName"].astype(str).map(normalize_city_key)
    return mentions_df


def annotate_ground_truth_detectability(
    ground_truth_rows: list[dict[str, Any]],
    mentions_df: pd.DataFrame,
    mentions_threshold: int = 100,
    radius_km: float = 50.0,
) -> list[dict[str, Any]]:
    working_df = mentions_df.copy()
    if "date" not in working_df.columns:
        working_df["date"] = working_df["SQLDATE"].astype(str).str[:8]
    if "normalized_city" not in working_df.columns:
        working_df["normalized_city"] = working_df["ActionGeo_FullName"].astype(str).map(normalize_city_key)

    annotated_rows: list[dict[str, Any]] = []

    for row in ground_truth_rows:
        same_day = working_df[working_df["date"] == row["date"]]
        mentions = 0
        match_type = "none"

        exact_hits = same_day[same_day["normalized_city"] == row["normalized_city"]]
        if not exact_hits.empty:
            mentions = int(exact_hits["NumMentions"].sum())
            match_type = "normalized"
        else:
            substring_hits = same_day[
                same_day["normalized_city"].astype(str).map(
                    lambda value: (
                        row["normalized_city"] in value
                        or value in row["normalized_city"]
                    )
                )
            ]
            if not substring_hits.empty:
                mentions = int(substring_hits["NumMentions"].sum())
                match_type = "substring"
            else:
                geo_hits = []
                for _, candidate in same_day.iterrows():
                    if pd.isna(candidate["ActionGeo_Lat"]) or pd.isna(candidate["ActionGeo_Long"]):
                        continue
                    distance = haversine_km(
                        float(row["lat"]),
                        float(row["lon"]),
                        float(candidate["ActionGeo_Lat"]),
                        float(candidate["ActionGeo_Long"]),
                    )
                    if distance <= radius_km:
                        geo_hits.append(int(candidate["NumMentions"]))
                if geo_hits:
                    mentions = int(sum(geo_hits))
                    match_type = "geo"

        annotated = row.copy()
        annotated["mentions_on_date"] = mentions
        annotated["tier"] = "A" if mentions >= mentions_threshold else "C"
        annotated["mentions_match_type"] = match_type
        annotated_rows.append(annotated)

    return annotated_rows


def _date_to_dt(date: str) -> datetime:
    return datetime.strptime(str(date), "%Y%m%d")


def _date_window(start_date: str, window_days: int, direction: str = "forward") -> set[str]:
    start_dt = _date_to_dt(start_date)
    if direction == "forward":
        return {
            (start_dt + timedelta(days=offset)).strftime("%Y%m%d")
            for offset in range(window_days + 1)
        }
    return {
        (start_dt - timedelta(days=offset)).strftime("%Y%m%d")
        for offset in range(window_days + 1)
    }


def _prediction_matches_ground_truth(
    prediction: dict[str, Any],
    ground_truth: dict[str, Any],
    radius_km: float = 50.0,
) -> tuple[bool, str | None, float | None]:
    if prediction["normalized_city"] == ground_truth["normalized_city"]:
        return True, "name", 0.0

    pred_lat = prediction.get("lat")
    pred_lon = prediction.get("lon")
    gt_lat = ground_truth.get("lat")
    gt_lon = ground_truth.get("lon")
    if None in (pred_lat, pred_lon, gt_lat, gt_lon):
        return False, None, None

    distance = haversine_km(float(pred_lat), float(pred_lon), float(gt_lat), float(gt_lon))
    if distance <= radius_km:
        return True, "geo", distance
    return False, None, None


def flatten_predictions(daily_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for result in daily_results:
        for prediction in result.get("predictions", []):
            enriched = prediction.copy()
            enriched["run_date"] = result["date"]
            predictions.append(enriched)
    return predictions


def evaluate_windowed_metrics(
    daily_results: list[dict[str, Any]],
    ground_truth_rows: list[dict[str, Any]],
    window_days: int = 7,
    radius_km: float = 50.0,
) -> dict[str, Any]:
    predictions = flatten_predictions(daily_results)
    predictions_by_date: dict[str, list[dict[str, Any]]] = {}
    for prediction in predictions:
        predictions_by_date.setdefault(prediction["run_date"], []).append(prediction)

    gt_hits = []
    gt_hit_keys: set[tuple[str, str]] = set()
    for ground_truth in ground_truth_rows:
        window_dates = _date_window(ground_truth["date"], window_days, direction="forward")
        matched_prediction = None
        matched_type = None
        matched_distance = None

        for date in sorted(window_dates):
            for prediction in predictions_by_date.get(date, []):
                matched, match_type, distance = _prediction_matches_ground_truth(
                    prediction,
                    ground_truth,
                    radius_km=radius_km,
                )
                if matched:
                    matched_prediction = prediction
                    matched_type = match_type
                    matched_distance = distance
                    break
            if matched_prediction is not None:
                break

        if matched_prediction is not None:
            gt_key = (ground_truth["date"], ground_truth["normalized_city"])
            gt_hit_keys.add(gt_key)
            gt_hits.append(
                {
                    "gt_date": ground_truth["date"],
                    "gt_city": ground_truth["city"],
                    "prediction_date": matched_prediction["run_date"],
                    "prediction_city": matched_prediction["city"],
                    "tier": ground_truth["tier"],
                    "match_type": matched_type,
                    "distance_km": None if matched_distance is None else round(float(matched_distance), 2),
                }
            )

    tp_predictions = []
    fp_predictions = []
    tp_count = 0
    tp_count_a = 0
    fp_count = 0

    for prediction in predictions:
        window_dates = _date_window(prediction["run_date"], window_days, direction="backward")
        candidate_ground_truth = [
            row
            for row in ground_truth_rows
            if row["date"] in window_dates
        ]

        matched_ground_truth = None
        matched_type = None
        matched_distance = None
        for ground_truth in candidate_ground_truth:
            matched, match_type, distance = _prediction_matches_ground_truth(
                prediction,
                ground_truth,
                radius_km=radius_km,
            )
            if matched:
                matched_ground_truth = ground_truth
                matched_type = match_type
                matched_distance = distance
                break

        if matched_ground_truth is None:
            fp_count += 1
            fp_predictions.append(
                {
                    "prediction_date": prediction["run_date"],
                    "prediction_city": prediction["city"],
                }
            )
            continue

        tp_count += 1
        if matched_ground_truth["tier"] == "A":
            tp_count_a += 1
        tp_predictions.append(
            {
                "prediction_date": prediction["run_date"],
                "prediction_city": prediction["city"],
                "gt_date": matched_ground_truth["date"],
                "gt_city": matched_ground_truth["city"],
                "tier": matched_ground_truth["tier"],
                "match_type": matched_type,
                "distance_km": None if matched_distance is None else round(float(matched_distance), 2),
            }
        )

    gt_count_total = len(ground_truth_rows)
    gt_rows_a = [row for row in ground_truth_rows if row["tier"] == "A"]
    gt_rows_c = [row for row in ground_truth_rows if row["tier"] == "C"]
    gt_keys_a = {(row["date"], row["normalized_city"]) for row in gt_rows_a}
    gt_keys_c = {(row["date"], row["normalized_city"]) for row in gt_rows_c}

    hit_count_total = len(gt_hit_keys)
    hit_count_a = len([key for key in gt_hit_keys if key in gt_keys_a])
    hit_count_c = len([key for key in gt_hit_keys if key in gt_keys_c])

    prediction_count_total = len(predictions)
    precision = tp_count / prediction_count_total if prediction_count_total else 0.0
    recall_total = hit_count_total / gt_count_total if gt_count_total else 0.0
    recall_a = hit_count_a / len(gt_rows_a) if gt_rows_a else 0.0
    recall_c = hit_count_c / len(gt_rows_c) if gt_rows_c else 0.0
    precision_a = tp_count_a / prediction_count_total if prediction_count_total else 0.0
    f1 = 2 * precision * recall_total / (precision + recall_total) if (precision + recall_total) else 0.0
    f1_a = 2 * precision_a * recall_a / (precision_a + recall_a) if (precision_a + recall_a) else 0.0

    return {
        "window_days": window_days,
        "prediction_count_total": prediction_count_total,
        "tp_count": tp_count,
        "tp_count_a": tp_count_a,
        "fp_count": fp_count,
        "gt_count_total": gt_count_total,
        "gt_count_a": len(gt_rows_a),
        "gt_count_c": len(gt_rows_c),
        "hit_count_total": hit_count_total,
        "hit_count_a": hit_count_a,
        "hit_count_c": hit_count_c,
        "precision": round(precision, 4),
        "precision_a": round(precision_a, 4),
        "recall_total": round(recall_total, 4),
        "recall_a": round(recall_a, 4),
        "recall_c": round(recall_c, 4),
        "f1": round(f1, 4),
        "f1_a": round(f1_a, 4),
        "gt_hits": gt_hits,
        "tp_predictions": tp_predictions,
        "fp_predictions": fp_predictions,
    }


def extract_predicted_cities(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    recommendations = schedule.get("city_best_recommendations") or schedule.get("recommendations") or []
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in recommendations:
        city = str(item.get("city", "")).strip()
        if not city:
            continue
        normalized_city = normalize_city_key(city)
        if normalized_city in seen:
            continue
        seen.add(normalized_city)
        deduped.append(
            {
                "city": city,
                "normalized_city": normalized_city,
                "lat": item.get("lat"),
                "lon": item.get("lon"),
                "satellite": item.get("satellite", ""),
                "sensor_type": item.get("sensor_type", ""),
                "priority_label": item.get("action_priority_label", ""),
                "pass_time_utc": item.get("pass_time_utc", ""),
            }
        )

    return deduped


def extract_level1_alert_cities(level1_output: dict[str, Any]) -> list[dict[str, Any]]:
    alerts = level1_output.get("alerts", []) or []
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()

    for alert in alerts:
        city = str(alert.get("city", "")).split(",")[0].strip()
        if not city:
            continue
        normalized_city = normalize_city_key(city)
        if normalized_city in seen:
            continue
        seen.add(normalized_city)
        deduped.append(
            {
                "city": city,
                "normalized_city": normalized_city,
                "lat": alert.get("lat"),
                "lon": alert.get("lon"),
                "risk_label": alert.get("risk_label", ""),
                "llm_status": alert.get("llm_status", "UNVERIFIED"),
            }
        )

    return deduped


def match_predictions_to_ground_truth(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    radius_km: float = 50.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    gt_by_key: dict[str, list[int]] = {}
    for idx, gt in enumerate(ground_truth):
        gt_by_key.setdefault(gt["normalized_city"], []).append(idx)

    unmatched_gt = set(range(len(ground_truth)))
    matched: list[dict[str, Any]] = []
    unmatched_predictions: list[dict[str, Any]] = []
    geo_candidates: list[dict[str, Any]] = []

    for prediction in predictions:
        match_idx = None
        for idx in gt_by_key.get(prediction["normalized_city"], []):
            if idx in unmatched_gt:
                match_idx = idx
                break

        if match_idx is not None:
            gt = ground_truth[match_idx]
            matched.append(
                {
                    "prediction_city": prediction["city"],
                    "ground_truth_city": gt["city"],
                    "match_type": "name",
                    "distance_km": 0.0,
                }
            )
            unmatched_gt.remove(match_idx)
            continue

        geo_candidates.append(prediction)

    for prediction in geo_candidates:
        lat = prediction.get("lat")
        lon = prediction.get("lon")
        if lat is None or lon is None:
            unmatched_predictions.append(prediction)
            continue

        best_idx = None
        best_distance = None
        for idx in sorted(unmatched_gt):
            gt = ground_truth[idx]
            distance = haversine_km(float(lat), float(lon), float(gt["lat"]), float(gt["lon"]))
            if distance > radius_km:
                continue
            if best_distance is None or distance < best_distance:
                best_idx = idx
                best_distance = distance

        if best_idx is None:
            unmatched_predictions.append(prediction)
            continue

        gt = ground_truth[best_idx]
        matched.append(
            {
                "prediction_city": prediction["city"],
                "ground_truth_city": gt["city"],
                "match_type": "geo",
                "distance_km": round(float(best_distance), 2),
            }
        )
        unmatched_gt.remove(best_idx)

    unmatched_ground_truth = [ground_truth[idx] for idx in sorted(unmatched_gt)]
    return matched, unmatched_predictions, unmatched_ground_truth


def available_gt_dates() -> list[str]:
    return sorted(
        path.stem
        for path in GROUND_TRUTH_DIR.glob("*.csv")
        if path.stem.isdigit() and len(path.stem) == 8
    )


def evaluate_single_date(
    date: str,
    scenario: str = "coverage",
    hours: int = 168,
    top_k: int = 20,
    radius_km: float = 50.0,
    use_llm: bool = False,
    quiet: bool = True,
) -> dict[str, Any]:
    buffer = io.StringIO()
    stdout_cm = contextlib.redirect_stdout(buffer) if quiet else contextlib.nullcontext()

    with stdout_cm:
        result = run_integrated_pipeline(
            target_date=date,
            hours=hours,
            use_llm=use_llm,
            top_k=top_k,
            mode="backtest",
            scenario=scenario,
            save_level1=False,
            save_schedule_output=False,
        )

    logs = buffer.getvalue()
    level1_output = result.get("level1_output", {})
    schedule = result.get("schedule", {})
    predictions = extract_predicted_cities(schedule)
    level1_predictions = extract_level1_alert_cities(level1_output)
    ground_truth = load_ground_truth_rows(date)
    matched, unmatched_predictions, unmatched_ground_truth = match_predictions_to_ground_truth(
        predictions,
        ground_truth,
        radius_km=radius_km,
    )

    tp = len(matched)
    prediction_count = len(predictions)
    gt_count = len(ground_truth)
    precision = tp / prediction_count if prediction_count else 0.0
    recall = tp / gt_count if gt_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    sensor_summary = schedule.get("sensor_condition_summary", {})
    alerts = level1_output.get("alerts", [])
    llm_status_counts: dict[str, int] = {}
    for alert in alerts:
        status = str(alert.get("llm_status", "UNVERIFIED") or "UNVERIFIED")
        llm_status_counts[status] = llm_status_counts.get(status, 0) + 1

    return {
        "date": date,
        "scenario": scenario,
        "mode": result.get("mode"),
        "used_llm": use_llm,
        "level1_alert_count": int(result.get("level1_output", {}).get("alert_count", 0)),
        "level1_predictions": level1_predictions,
        "gt_count": gt_count,
        "predicted_city_count": prediction_count,
        "tp": tp,
        "fp": len(unmatched_predictions),
        "fn": len(unmatched_ground_truth),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "weather_request_failed": "[WEATHER] ⚠️ Open-Meteo 요청 실패" in logs,
        "eo_candidates": int(sensor_summary.get("optical_total", 0)),
        "eo_shootable": int(sensor_summary.get("optical_shootable", 0)),
        "eo_weather_unknown": int(sensor_summary.get("optical_blocked_unknown", 0)),
        "sar_candidates": int(sensor_summary.get("sar_total", 0)),
        "sar_shootable": int(sensor_summary.get("sar_shootable", 0)),
        "llm_candidate_count": len(alerts),
        "llm_confirmed_count": llm_status_counts.get("SUCCESS", 0) + llm_status_counts.get("AMBIGUOUS", 0),
        "llm_rejected_count": sum(
            llm_status_counts.get(status, 0)
            for status in ("DROPPED", "NO_MENTION", "DATE_MISMATCH")
        ),
        "llm_error_count": sum(
            llm_status_counts.get(status, 0)
            for status in ("ERROR", "ARTICLE_UNREACHABLE")
        ),
        "llm_status_counts": llm_status_counts,
        "predictions": predictions,
        "matched_pairs": matched,
        "unmatched_predictions": unmatched_predictions,
        "unmatched_ground_truth": unmatched_ground_truth,
        "schedule_summary": {
            "total_passes": schedule.get("total_passes", 0),
            "swath_passes": schedule.get("swath_passes", 0),
            "shootable_passes": schedule.get("shootable_passes", 0),
            "scheduled_cities": schedule.get("scheduled_cities", 0),
            "satellites_used": schedule.get("satellites_used", 0),
        },
    }


def _daily_result_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": result["date"],
        "scenario": result["scenario"],
        "mode": result["mode"],
        "used_llm": result["used_llm"],
        "level1_alert_count": result["level1_alert_count"],
        "predicted_city_count": result["predicted_city_count"],
        "weather_request_failed": result["weather_request_failed"],
        "eo_candidates": result["eo_candidates"],
        "eo_shootable": result["eo_shootable"],
        "eo_weather_unknown": result["eo_weather_unknown"],
        "sar_candidates": result["sar_candidates"],
        "sar_shootable": result["sar_shootable"],
        "llm_candidate_count": result["llm_candidate_count"],
        "llm_confirmed_count": result["llm_confirmed_count"],
        "llm_rejected_count": result["llm_rejected_count"],
        "llm_error_count": result["llm_error_count"],
        "total_passes": result["schedule_summary"]["total_passes"],
        "swath_passes": result["schedule_summary"]["swath_passes"],
        "shootable_passes": result["schedule_summary"]["shootable_passes"],
        "scheduled_cities": result["schedule_summary"]["scheduled_cities"],
        "satellites_used": result["schedule_summary"]["satellites_used"],
    }


def _aggregate_llm_status_counts(daily_results: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for result in daily_results:
        counts.update(result.get("llm_status_counts", {}))
    return dict(sorted(counts.items()))


def save_results(
    daily_results: list[dict[str, Any]],
    ground_truth_rows: list[dict[str, Any]],
    windowed_metrics: dict[str, Any],
    scenario: str,
    use_llm: bool = False,
    window_days: int = 7,
    mentions_threshold: int = 100,
) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_llm" if use_llm else ""
    json_path = OUTPUT_DIR / f"gt_eval_windowed_{scenario}{suffix}.json"
    csv_path = OUTPUT_DIR / f"gt_eval_windowed_{scenario}{suffix}.csv"

    daily_rows = [_daily_result_row(result) for result in daily_results]
    summary_row = {
        "scenario": scenario,
        "used_llm": use_llm,
        "window_days": window_days,
        "mentions_threshold": mentions_threshold,
        "gt_count_total": windowed_metrics["gt_count_total"],
        "gt_count_a": windowed_metrics["gt_count_a"],
        "gt_count_c": windowed_metrics["gt_count_c"],
        "prediction_count_total": windowed_metrics["prediction_count_total"],
        "tp_count": windowed_metrics["tp_count"],
        "tp_count_a": windowed_metrics["tp_count_a"],
        "fp_count": windowed_metrics["fp_count"],
        "precision": windowed_metrics["precision"],
        "precision_a": windowed_metrics["precision_a"],
        "recall_total": windowed_metrics["recall_total"],
        "recall_a": windowed_metrics["recall_a"],
        "recall_c": windowed_metrics["recall_c"],
        "f1": windowed_metrics["f1"],
        "f1_a": windowed_metrics["f1_a"],
    }

    json_path.write_text(
        json.dumps(
            {
                "scenario": scenario,
                "used_llm": use_llm,
                "date_count": len(daily_results),
                "window_days": window_days,
                "mentions_threshold": mentions_threshold,
                "llm_status_counts": _aggregate_llm_status_counts(daily_results),
                "windowed_summary": summary_row,
                "ground_truth_rows": ground_truth_rows,
                "daily_results": daily_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    return json_path, csv_path


def print_daily_run_table(daily_results: list[dict[str, Any]]) -> None:
    print()
    print(
        f"{'date':10s} | {'alerts':>6s} | {'pred':>4s} | {'EO':>7s} | {'SAR':>7s} | {'LLM':>7s}"
    )
    print("-" * 62)
    for result in daily_results:
        print(
            f"{result['date']:10s} | "
            f"{result['level1_alert_count']:>6d} | "
            f"{result['predicted_city_count']:>4d} | "
            f"{result['eo_shootable']:>2d}/{result['eo_candidates']:<4d} | "
            f"{result['sar_shootable']:>2d}/{result['sar_candidates']:<4d} | "
            f"{result['llm_confirmed_count']:>2d}/{result['llm_candidate_count']:<4d}"
        )

    print("-" * 62)


def print_windowed_summary(windowed_metrics: dict[str, Any], mentions_threshold: int, window_days: int) -> None:
    print()
    print(f"Window: +{window_days}d | Tier threshold: mentions_on_date >= {mentions_threshold}")
    print(
        f"GT total={windowed_metrics['gt_count_total']} | "
        f"A={windowed_metrics['gt_count_a']} | "
        f"C={windowed_metrics['gt_count_c']}"
    )
    print(
        f"Pred total={windowed_metrics['prediction_count_total']} | "
        f"TP={windowed_metrics['tp_count']} | "
        f"FP={windowed_metrics['fp_count']}"
    )
    print(
        f"Precision(all)={windowed_metrics['precision']:.4f} | "
        f"Recall(all)={windowed_metrics['recall_total']:.4f} | "
        f"F1(all)={windowed_metrics['f1']:.4f}"
    )
    print(
        f"Precision(A)={windowed_metrics['precision_a']:.4f} | "
        f"Recall(A)={windowed_metrics['recall_a']:.4f} | "
        f"Recall(C)={windowed_metrics['recall_c']:.4f} | "
        f"F1(A)={windowed_metrics['f1_a']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrated pipeline Ground Truth evaluator")
    parser.add_argument("--scenario", default="coverage", help="Satellite scenario")
    parser.add_argument("--dates", nargs="*", help="Target dates (YYYYMMDD). Default: all ground-truth dates")
    parser.add_argument("--hours", type=int, default=168, help="Prediction window hours")
    parser.add_argument("--top-k", type=int, default=20, help="Level1 candidate upper bound")
    parser.add_argument("--radius-km", type=float, default=50.0, help="Geo match radius")
    parser.add_argument("--window-days", type=int, default=7, help="TP credit window in days")
    parser.add_argument("--mentions-threshold", type=int, default=100, help="Tier A threshold for mentions_on_date")
    parser.add_argument("--use-llm", action="store_true", help="Enable Level1 LLM verification")
    parser.add_argument("--verbose", action="store_true", help="Show integrated pipeline stdout")
    args = parser.parse_args()

    dates = args.dates or available_gt_dates()
    daily_results = []

    for index, date in enumerate(dates, 1):
        print(f"[{index}/{len(dates)}] {date} evaluating...")
        daily_results.append(
            evaluate_single_date(
                date=date,
                scenario=args.scenario,
                hours=args.hours,
                top_k=args.top_k,
                radius_km=args.radius_km,
                use_llm=args.use_llm,
                quiet=not args.verbose,
            )
        )

    ground_truth_rows = load_all_ground_truth_rows(dates)
    mentions_df = load_mentions_frame()
    annotated_ground_truth_rows = annotate_ground_truth_detectability(
        ground_truth_rows,
        mentions_df,
        mentions_threshold=args.mentions_threshold,
        radius_km=args.radius_km,
    )
    windowed_metrics = evaluate_windowed_metrics(
        daily_results,
        annotated_ground_truth_rows,
        window_days=args.window_days,
        radius_km=args.radius_km,
    )

    print_daily_run_table(daily_results)
    print_windowed_summary(
        windowed_metrics,
        mentions_threshold=args.mentions_threshold,
        window_days=args.window_days,
    )
    json_path, csv_path = save_results(
        daily_results,
        annotated_ground_truth_rows,
        windowed_metrics,
        args.scenario,
        use_llm=args.use_llm,
        window_days=args.window_days,
        mentions_threshold=args.mentions_threshold,
    )
    print()
    print(f"saved json: {json_path}")
    print(f"saved csv : {csv_path}")


if __name__ == "__main__":
    main()
