"""
Unified Level1+2b -> Level2a pipeline runner.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.config import OUTPUT_DIR, PREDICTION_HOURS, OPERATIONAL_SATELLITE_SCENARIO


REPO_ROOT = Path(__file__).resolve().parent.parent
LEVEL1_ROOT = REPO_ROOT / "pipeline" / "Level1+2b"
LLM_PASS_STATUSES = {"SUCCESS", "AMBIGUOUS"}
LEVEL2A_RISK_BY_LEVEL = {
    3: "RED",
    2: "ORANGE",
    1: "YELLOW",
    0: "BLUE",
}
LEVEL2A_RISK_BY_LABEL = {
    "위기": "RED",
    "위험": "ORANGE",
    "주의": "YELLOW",
    "정상": "BLUE",
    "RED": "RED",
    "ORANGE": "ORANGE",
    "YELLOW": "YELLOW",
    "BLUE": "BLUE",
}


def _resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_main_data_path() -> Path:
    return _resolve_existing_path(
        REPO_ROOT / "gdelt_main_2026.parquet",
        REPO_ROOT / "data" / "gdelt_main_2026.parquet",
        REPO_ROOT / "gdelt_main_final.parquet",
        REPO_ROOT / "data" / "gdelt_main_final.parquet",
    )


def resolve_url_data_path() -> Path:
    return _resolve_existing_path(
        REPO_ROOT / "gdelt_url_final.parquet",
        REPO_ROOT / "data" / "gdelt_url_final.parquet",
        REPO_ROOT / "gdelt_url_2026.parquet",
        REPO_ROOT / "data" / "gdelt_url_2026.parquet",
    )


def resolve_level2a_mode(target_date: str, mode: str = "auto") -> str:
    if mode in {"operational", "backtest"}:
        return mode

    today_utc = datetime.now(timezone.utc).strftime("%Y%m%d")
    return "backtest" if str(target_date) < today_utc else "operational"


def _ensure_level1_path() -> None:
    level1_path = str(LEVEL1_ROOT)
    if level1_path not in sys.path:
        sys.path.insert(0, level1_path)


def _load_level1_modules():
    _ensure_level1_path()
    try:
        from src.kalman_filter import compute_conflict_index, detect_anomalies
        import src.preprocess as preprocess_module
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Level1+2b dependencies are missing. "
            "Install preprocessing/Kalman requirements before running the integrated pipeline."
        ) from exc

    return compute_conflict_index, detect_anomalies, preprocess_module


def _load_level1_llm_module():
    _ensure_level1_path()
    try:
        from src.llm_verification import verify_anomalies_with_llm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LLM verification dependencies are missing. "
            "Install Level1+2b requirements or run without --use-llm."
        ) from exc

    return verify_anomalies_with_llm


def _load_level2a_modules():
    try:
        from pipeline.level2a import build_schedule_from_level1_result
        from pipeline.schedule_builder import print_schedule, save_schedule
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Level2a dependencies are missing. "
            "Install orbit/weather packages before running the integrated pipeline."
        ) from exc

    return build_schedule_from_level1_result, print_schedule, save_schedule


def load_gdelt_datasets(
    target_date: str,
    main_path: str | None = None,
    url_path: str | None = None,
    load_url_data: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None, Path, Path | None]:
    main_data_path = Path(main_path) if main_path else resolve_main_data_path()
    raw_df = pd.read_parquet(main_data_path)
    raw_df["SQLDATE"] = raw_df["SQLDATE"].astype(str).str[:8]
    raw_df = raw_df[raw_df["SQLDATE"] <= str(target_date)].copy()

    url_df = None
    url_data_path = None
    if load_url_data:
        url_data_path = Path(url_path) if url_path else resolve_url_data_path()
        if url_data_path.exists():
            url_df = pd.read_parquet(url_data_path)

    return raw_df, url_df, main_data_path, url_data_path


def _extract_llm_summary(report_raw: Any) -> str:
    if not isinstance(report_raw, str) or not report_raw.strip():
        return ""
    try:
        parsed = json.loads(report_raw)
    except json.JSONDecodeError:
        return report_raw
    return str(parsed.get("Summary", "")).strip()


def _normalize_source_urls(source_urls_raw: Any, limit: int = 2) -> list[str]:
    if isinstance(source_urls_raw, list):
        values = source_urls_raw
    elif isinstance(source_urls_raw, str) and source_urls_raw.strip():
        values = [source_urls_raw]
    else:
        return []

    normalized = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        normalized.append(text)
        if len(normalized) >= limit:
            break
    return normalized


def map_level1_risk_to_level2a(row: pd.Series) -> str:
    risk_level = row.get("risk_level")
    if pd.notna(risk_level):
        try:
            mapped = LEVEL2A_RISK_BY_LEVEL.get(int(risk_level))
            if mapped:
                return mapped
        except (TypeError, ValueError):
            pass

    risk_label = str(row.get("risk_label", "")).strip()
    return LEVEL2A_RISK_BY_LABEL.get(risk_label, "YELLOW")


def build_level1_output(
    anomalies: pd.DataFrame,
    target_date: str,
    use_llm: bool = False,
    top_k: int = 20,
) -> dict[str, Any]:
    if anomalies.empty:
        return {
            "date": target_date,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "alert_count": 0,
            "alerts": [],
        }

    selected = anomalies.copy()
    selected["date"] = selected["date"].astype(str)
    selected = selected[(selected["date"] == str(target_date)) & (selected["is_anomaly"] == True)].copy()

    if use_llm and "llm_status" in selected.columns:
        selected = selected[selected["llm_status"].isin(LLM_PASS_STATUSES)].copy()

    if selected.empty:
        return {
            "date": target_date,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "alert_count": 0,
            "alerts": [],
        }

    sort_cols = [col for col in ["innov_z", "conflict_index", "events"] if col in selected.columns]
    if sort_cols:
        selected = selected.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    selected = selected.head(top_k)

    alerts = []
    for _, row in selected.iterrows():
        lon = row.get("lng", row.get("lon"))
        alert = {
            "city": row["city"],
            "risk_level": int(row["risk_level"]) if pd.notna(row.get("risk_level")) else None,
            "risk_label": map_level1_risk_to_level2a(row),
            "conflict_index": round(float(row.get("conflict_index", 0.0)), 3),
            "innovation_z": round(float(row.get("innov_z", 0.0)), 3),
            "severity_score": round(float(row.get("innov_z", 0.0)), 3),
            "events": int(row.get("events", 0) or 0),
            "lat": None if pd.isna(row.get("lat")) else round(float(row.get("lat")), 4),
            "lon": None if pd.isna(lon) else round(float(lon), 4),
            "country_code": str(row.get("country_code", "") or ""),
            "guide": str(row.get("risk_guide", "") or ""),
            "llm_status": str(row.get("llm_status", "UNVERIFIED") or "UNVERIFIED"),
            "llm_event_summary": _extract_llm_summary(row.get("llm_report")),
            "source_urls": _normalize_source_urls(row.get("source_urls")),
        }
        alerts.append(alert)

    return {
        "date": target_date,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "alert_count": len(alerts),
        "alerts": alerts,
    }


def save_level1_output(level1_output: dict[str, Any], target_date: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{target_date}.json"
    output_path.write_text(json.dumps(level1_output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def print_level1_summary(level1_output: dict[str, Any], use_llm: bool) -> None:
    print("\n" + "═" * 85)
    print("  📡 Level 1+2b 후보 도시 요약")
    print(f"  📅 대상 날짜: {level1_output.get('date')}")
    print(f"  🤖 LLM 검증: {'ON' if use_llm else 'OFF'}")
    print(f"  🚨 후보 도시 수: {level1_output.get('alert_count', 0)}")
    print("═" * 85)

    alerts = level1_output.get("alerts", [])
    if not alerts:
        print("  ⚠️ 위성 스케줄링으로 넘길 후보 도시가 없습니다.\n")
        return

    for rank, alert in enumerate(alerts, 1):
        city = str(alert.get("city", "")).split(",")[0]
        print(
            f"  {rank:>2d}. {city:15s} | {alert.get('risk_label', 'N/A'):7s} | "
            f"Z={float(alert.get('innovation_z', 0.0)):>6.2f} | "
            f"{alert.get('llm_status', 'UNVERIFIED')}"
        )
    print()


def _stdout_sink(verbose: bool):
    return contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())


def run_integrated_pipeline(
    target_date: str,
    hours: int = PREDICTION_HOURS,
    use_llm: bool = False,
    top_k: int = 20,
    mode: str = "auto",
    tle_date: str | None = None,
    refresh: bool = False,
    scenario: str = OPERATIONAL_SATELLITE_SCENARIO,
    save_level1: bool = True,
    save_schedule_output: bool = True,
    main_path: str | None = None,
    url_path: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    compute_conflict_index, detect_anomalies, preprocess_module = _load_level1_modules()
    preprocess_module.VERBOSE_LOGS = verbose
    raw_df, url_df, main_data_path, url_data_path = load_gdelt_datasets(
        target_date=target_date,
        main_path=main_path,
        url_path=url_path,
        load_url_data=use_llm,
    )

    print("\n" + "═" * 85)
    print("  🛰️  SIA 통합 파이프라인")
    print(f"  📅 대상 날짜: {target_date}")
    print(f"  🗂️ GDELT 본문: {main_data_path.name}")
    if use_llm and url_data_path is not None:
        print(f"  🔗 URL 매핑: {url_data_path.name}")
    print("═" * 85)

    with _stdout_sink(verbose):
        city_daily, filtered_df = compute_conflict_index(raw_df, is_train=False)
    anomalies_today = detect_anomalies(city_daily, target_date)
    print(f"  [L1] 칼만 필터 통과 후보: {len(anomalies_today)}건")

    anomalies_for_schedule = anomalies_today
    if use_llm:
        if url_df is None:
            raise RuntimeError("LLM verification requested, but URL parquet could not be loaded.")
        verify_anomalies_with_llm = _load_level1_llm_module()
        with _stdout_sink(verbose):
            anomalies_for_schedule = verify_anomalies_with_llm(
                anomalies_today.copy(),
                filtered_df,
                url_df,
                target_date,
                top_k=top_k,
            )

    level1_output = build_level1_output(
        anomalies_for_schedule,
        target_date=target_date,
        use_llm=use_llm,
        top_k=top_k,
    )
    print_level1_summary(level1_output, use_llm=use_llm)

    level1_output_path = None
    if save_level1:
        level1_output_path = save_level1_output(level1_output, target_date)
        print(f"  [L1] 저장: {level1_output_path}")

    if not level1_output["alerts"]:
        return {
            "level1_output": level1_output,
            "level1_output_path": str(level1_output_path) if level1_output_path else None,
            "schedule": {"error": "스케줄링 가능한 후보 도시가 없습니다.", "alert_count": 0},
            "schedule_path": None,
        }

    build_schedule_from_level1_result, print_schedule, save_schedule = _load_level2a_modules()
    resolved_mode = resolve_level2a_mode(target_date, mode=mode)
    with _stdout_sink(verbose):
        schedule = build_schedule_from_level1_result(
            level1_data=level1_output,
            target_date=target_date,
            hours=hours,
            mode=resolved_mode,
            tle_date=tle_date,
            refresh=refresh,
            scenario=scenario,
        )

    if "error" in schedule:
        return {
            "level1_output": level1_output,
            "level1_output_path": str(level1_output_path) if level1_output_path else None,
            "schedule": schedule,
            "schedule_path": None,
        }

    print_schedule(schedule, verbose=verbose)

    schedule_path = None
    if save_schedule_output:
        suffix = "backtest" if resolved_mode == "backtest" else "real"
        scenario_suffix = "" if scenario == "default" else f"_{scenario}"
        schedule_path = save_schedule(
            schedule,
            filename=f"schedule_{target_date}_{suffix}{scenario_suffix}.json",
        )

    return {
        "level1_output": level1_output,
        "level1_output_path": str(level1_output_path) if level1_output_path else None,
        "schedule": schedule,
        "schedule_path": str(schedule_path) if schedule_path else None,
        "mode": resolved_mode,
    }
