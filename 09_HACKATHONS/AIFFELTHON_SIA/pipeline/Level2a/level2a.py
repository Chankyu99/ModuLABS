"""
Level 2a — 통합 진입점
────────────────────────────────────
Level 1 결과(JSON)를 받아 Level 2a 입력으로 정규화하고,
촬영 스케줄 생성/출력/저장을 한 곳에서 다룬다.

의도:
- Level 2a 관련 엔트리 로직을 한 파일에 모아 유지보수 포인트를 줄인다.
- 세부 계산 모듈(pass/weather/tle/schedule_builder)은 그대로 두고,
  실제 운용 코드가 만지는 인터페이스만 얇게 통합한다.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import OUTPUT_DIR, ROI_CITIES, PREDICTION_HOURS
from pipeline.schedule_builder import build_schedule, print_schedule, save_schedule


def load_level1_results(date_str: str) -> dict | None:
    """해당 날짜의 Level 1 output JSON을 로드한다."""
    json_path = OUTPUT_DIR / f"{date_str}.json"
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_city_coordinates(city: str, alert: dict) -> tuple[dict | None, str]:
    """Level 1 좌표를 우선 사용하고, 없으면 정적 ROI를 fallback으로 사용한다."""
    lat = alert.get("lat")
    lon = alert.get("lon")

    if lat is not None and lon is not None:
        try:
            return {"lat": float(lat), "lon": float(lon)}, "level1"
        except (TypeError, ValueError):
            pass

    if city in ROI_CITIES:
        return ROI_CITIES[city], "roi"

    return None, "missing"


def extract_level2a_inputs(level1_data: dict) -> tuple[dict, dict]:
    """Level 1 결과에서 Level 2a용 도시 좌표와 위험 메타를 추출한다."""
    alerts = level1_data.get("alerts", [])
    risk_cities = {}
    valid_cities = {}

    print("  [Level 1] 다음 도시에 대한 위기 경보 접수:")
    for alert in alerts:
        city_full = alert["city"]
        city = city_full.split(",")[0].strip()
        risk_label = alert["risk_label"].split("(")[-1].strip(")")

        coords, coord_source = resolve_city_coordinates(city, alert)
        if coords is None:
            print(f"    ⚠️ {city} 좌표 미확보 (Level 1/ROI 모두 없음)")
            continue

        risk_cities[city] = {
            "risk_label": risk_label,
            "innovation_z": alert.get("innovation_z", 0.0),
            "severity_score": alert.get("innovation_z", 0.0),
            "lat": coords["lat"],
            "lon": coords["lon"],
            "country_code": alert.get("country_code", ""),
            "guide": alert.get("guide", ""),
            "llm_status": alert.get("llm_status", "UNVERIFIED"),
            "llm_event_summary": alert.get("llm_event_summary", ""),
            "source_urls": alert.get("source_urls", []),
        }
        valid_cities[city] = coords

        emoji = {"RED": "🛑", "ORANGE": "🟠", "YELLOW": "🟡"}.get(risk_label, "⚪")
        source_tag = "L1" if coord_source == "level1" else "ROI"
        print(
            f"    {emoji} {city:15s} ({risk_label:6s}) - Z-score: {alert['innovation_z']} "
            f"[coord:{source_tag}]"
        )

    return valid_cities, risk_cities


def resolve_prediction_context(
    target_date: str,
    mode: str = "operational",
    tle_date: str | None = None,
) -> tuple[datetime, str]:
    """실행 모드에 맞는 예측 시작 시각과 TLE 기준일을 계산한다."""
    if mode == "backtest":
        prediction_start_utc = datetime.strptime(target_date, "%Y%m%d").replace(
            tzinfo=timezone.utc
        )
        tle_reference_date = tle_date or target_date
        return prediction_start_utc, tle_reference_date

    prediction_start_utc = datetime.now(timezone.utc)
    tle_reference_date = tle_date or prediction_start_utc.strftime("%Y%m%d")
    return prediction_start_utc, tle_reference_date


def build_schedule_from_level1_result(
    level1_data: dict,
    target_date: str,
    hours: int = PREDICTION_HOURS,
    mode: str = "operational",
    tle_date: str | None = None,
    refresh: bool = False,
    scenario: str = "default",
) -> dict:
    """Level 1 결과를 입력으로 Level 2a 스케줄을 생성한다."""
    valid_cities, risk_cities = extract_level2a_inputs(level1_data)

    if not valid_cities:
        return {"error": "스케줄링 가능한 ROI 대상 도시가 없습니다."}

    prediction_start_utc, tle_reference_date = resolve_prediction_context(
        target_date=target_date,
        mode=mode,
        tle_date=tle_date,
    )

    return build_schedule(
        cities=valid_cities,
        risk_cities=risk_cities,
        hours=hours,
        force_tle_refresh=refresh,
        tle_mode=mode,
        tle_reference_date=tle_reference_date,
        prediction_start_utc=prediction_start_utc,
        satellite_scenario=scenario,
    )


def run_level2a_for_date(
    target_date: str,
    hours: int = PREDICTION_HOURS,
    mode: str = "operational",
    tle_date: str | None = None,
    refresh: bool = False,
    scenario: str = "default",
    save_output: bool = True,
) -> dict:
    """날짜 기준으로 Level 1 결과를 읽어 Level 2a를 실행한다."""
    level1_data = load_level1_results(target_date)
    if not level1_data:
        return {
            "error": f"{target_date}의 Level 1 분석 결과를 찾을 수 없습니다.",
            "missing_path": str(OUTPUT_DIR / f"{target_date}.json"),
        }

    alerts = level1_data.get("alerts", [])
    if not alerts:
        return {
            "error": f"{target_date}에는 탐지된 이상 징후가 없습니다.",
            "alert_count": 0,
        }

    schedule = build_schedule_from_level1_result(
        level1_data=level1_data,
        target_date=target_date,
        hours=hours,
        mode=mode,
        tle_date=tle_date,
        refresh=refresh,
        scenario=scenario,
    )

    if "error" in schedule:
        return schedule

    print_schedule(schedule)

    if save_output:
        suffix = "backtest" if mode == "backtest" else "real"
        scenario_suffix = "" if scenario == "default" else f"_{scenario}"
        save_schedule(
            schedule,
            filename=f"schedule_{target_date}_{suffix}{scenario_suffix}.json",
        )

    return schedule
