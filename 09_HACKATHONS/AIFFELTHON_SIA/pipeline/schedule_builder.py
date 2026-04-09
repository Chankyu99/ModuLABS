"""
Level 2a — 촬영 스케줄 빌더
────────────────────────────────────
Level 1 갈등 탐지 결과 + 위성 통과 예측 + 기상 조건을 통합하여
최종 촬영 스케줄 데이터셋(JSON)을 생성한다.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import SATELLITES, ROI_CITIES, OUTPUT_DIR
from pipeline.tle_fetcher import load_all_tle
from pipeline.pass_predictor import predict_passes, filter_shootable
from pipeline.weather_checker import check_weather


def compute_priority_score(event: dict) -> float:
    """촬영 우선순위 점수를 계산한다.

    Priority = risk_weight × (1 - cloud/100) × (elevation/90)
    """
    # 리스크 가중치 (Level 1 결과와 연동 시 사용)
    risk_weights = {"RED": 3, "ORANGE": 2, "YELLOW": 1, "BLUE": 0.5, "N/A": 0.5}
    risk = event.get("risk_level", "N/A")
    risk_w = risk_weights.get(risk, 0.5)

    cloud = max(event.get("cloud_cover_pct", 50), 0)
    elev = max(event.get("max_elevation_deg", 10), 1)

    return round(risk_w * (1 - cloud / 100) * (elev / 90), 3)


def build_schedule(
    cities: dict | None = None,
    risk_cities: dict | None = None,
    hours: int = 48,
    force_tle_refresh: bool = False,
) -> dict:
    """촬영 스케줄을 생성한다.

    Args:
        cities: ROI 도시 좌표. None이면 config 기본값
        risk_cities: Level 1 결과에서 도시별 리스크 등급.
                     예: {"Isfahan": "RED", "Tehran": "YELLOW"}
        hours: 예측 범위(시간)
        force_tle_refresh: TLE 캐시 강제 갱신 여부

    Returns:
        스케줄 JSON 데이터 (dict)
    """
    if cities is None:
        cities = ROI_CITIES
    if risk_cities is None:
        risk_cities = {}

    now_utc = datetime.now(timezone.utc)

    # ── Step 1: TLE 수집 ──
    print("\n  ── Level 2a: 촬영 스케줄 빌더 가동 ──\n")
    tle_data = load_all_tle(force_refresh=force_tle_refresh)

    if not tle_data:
        print("  [SCHEDULE] ❌ TLE 데이터 없음. 종료.")
        return {"error": "TLE 데이터 수집 실패"}

    # ── Step 2: 궤도 통과 예측 ──
    print(f"\n  [PASS] {len(cities)}개 도시 × {len(tle_data)}개 위성 통과 예측 중...")
    all_passes = predict_passes(tle_data, cities=cities, hours=hours)
    shootable_passes = filter_shootable(all_passes)
    print(f"  [PASS] 전체 통과: {len(all_passes)}건 | Swath 내: {len(shootable_passes)}건")

    # ── Step 3: 기상 판별 ──
    print(f"\n  [WEATHER] {len(shootable_passes)}건에 대해 기상 조건 확인 중...")
    weather_checked = []
    for p in shootable_passes:
        checked = check_weather(p)
        checked["risk_level"] = risk_cities.get(p["city"], "N/A")
        checked["priority_score"] = compute_priority_score(checked)
        weather_checked.append(checked)

    # ── Step 4: 촬영 가능 필터링 + 정렬 ──
    final_shootable = [e for e in weather_checked if e["shootable"]]
    final_shootable.sort(key=lambda x: x["priority_score"], reverse=True)

    # 위성 우선순위 적용 (동일 도시·시간대에서 SpaceEye-T 우선)
    final_shootable.sort(
        key=lambda x: (x["city"], x["pass_time_utc"], x["priority"])
    )

    # ── Step 5: 결과 조립 ──
    schedule = {
        "generated_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prediction_hours": hours,
        "satellites_tracked": len(tle_data),
        "cities_monitored": len(cities),
        "total_passes": len(all_passes),
        "swath_passes": len(shootable_passes),
        "shootable_passes": len(final_shootable),
        "recommendations": final_shootable,
    }

    return schedule


def save_schedule(schedule: dict, filename: str | None = None) -> Path:
    """스케줄을 JSON 파일로 저장한다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"schedule_{date_str}.json"

    filepath = OUTPUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2, ensure_ascii=False)

    print(f"\n  [SCHEDULE] 📄 저장: {filepath}")
    return filepath


def print_schedule(schedule: dict) -> None:
    """스케줄 리포트를 터미널에 출력한다."""
    recs = schedule.get("recommendations", [])

    print("\n" + "═" * 85)
    print("  🛰️  SIA 위성 촬영 스케줄 리포트")
    print(f"  📅 생성 시각: {schedule.get('generated_utc', 'N/A')}")
    print(f"  🔭 추적 위성: {schedule.get('satellites_tracked', 0)}개")
    print(f"  🏙️  모니터링 도시: {schedule.get('cities_monitored', 0)}개")
    print("═" * 85)

    print(f"\n  📊 통과 요약: 전체 {schedule.get('total_passes', 0)}건 "
          f"→ Swath 내 {schedule.get('swath_passes', 0)}건 "
          f"→ 촬영 가능 {schedule.get('shootable_passes', 0)}건\n")

    if not recs:
        print("  ⚠️ 촬영 가능한 통과 이벤트가 없습니다.\n")
        return

    print(f"  {'위성':12s} | {'도시':15s} | {'통과 시각(UTC)':22s} | "
          f"{'앙각':>5s} | {'구름':>4s} | {'주야':4s} | {'센서':4s} | {'리스크':8s}")
    print("  " + "─" * 83)

    for rec in recs[:30]:
        daylight_str = "☀️" if rec.get("daylight") else "🌙"
        sensor_str = "EO" if rec["sensor_type"] == "optical" else "SAR"
        cloud_str = f"{rec.get('cloud_cover_pct', '?')}%"
        risk_str = rec.get("risk_level", "N/A")

        print(
            f"  {rec['satellite']:12s} | {rec['city']:15s} | "
            f"{rec['pass_time_utc']:22s} | "
            f"{rec['max_elevation_deg']:5.1f}° | "
            f"{cloud_str:>4s} | {daylight_str:4s} | {sensor_str:4s} | {risk_str:8s}"
        )

    if len(recs) > 30:
        print(f"\n  ... 외 {len(recs) - 30}건")

    print()


# ──────────────────────────────────────────────
# CLI 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="촬영 스케줄 빌더")
    parser.add_argument("--hours", type=int, default=48, help="예측 범위(시간)")
    parser.add_argument("--refresh", action="store_true", help="TLE 캐시 강제 갱신")
    parser.add_argument("--save", action="store_true", help="JSON 파일로 저장")
    args = parser.parse_args()

    schedule = build_schedule(hours=args.hours, force_tle_refresh=args.refresh)

    if "error" not in schedule:
        print_schedule(schedule)
        if args.save:
            save_schedule(schedule)
