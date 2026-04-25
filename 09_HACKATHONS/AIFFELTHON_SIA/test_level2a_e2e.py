#!/usr/bin/env python3
"""
Level 1 → Level 2a 통합 시뮬레이션 (4월 3일 기준)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Level 1에서 4월 3일자 갈등 탐지가 완료되었다고 가정하고,
Level 2a 위성 촬영 스케줄을 자동 생성하는 전체 흐름 시뮬레이션.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── 프로젝트 루트를 PYTHONPATH에 추가 ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import ROI_CITIES, OUTPUT_DIR
from pipeline.tle_fetcher import load_all_tle
from pipeline.pass_predictor import predict_passes, filter_shootable
from pipeline.weather_checker import check_weather
from pipeline.schedule_builder import (
    build_schedule, save_schedule, print_schedule, compute_priority_score
)


def main():
    print("\n" + "━" * 85)
    print("  🔗 Level 1 → Level 2a 통합 시뮬레이션")
    print("  📅 시나리오: 2026년 4월 3일 데이터 업데이트 완료")
    print("━" * 85)

    # ──────────────────────────────────────────────
    # STEP 0: Level 1 탐지 결과 시뮬레이션
    # ──────────────────────────────────────────────
    # 4월 3일 Level 1 파이프라인 결과 (칼만 필터 이상 징후 탐지 → LLM 검증 완료)
    # 실제 운영에서는 run_daily.py의 output JSON을 읽어옴
    print("\n  ── STEP 0: Level 1 갈등 탐지 결과 (시뮬레이션) ──\n")

    level1_alerts = {
        "date": "20260403",
        "alerts": [
            {"city": "Isfahan",    "risk_level": "RED",    "z_score": 332.1,
             "events": 45, "guide": "대규모 공격 포착. 즉시 위성 촬영 스케줄링 필수."},
            {"city": "Tehran",     "risk_level": "RED",    "z_score": 186.5,
             "events": 38, "guide": "대규모 충돌/공격 포착. 즉시 위성 촬영 스케줄링 필수."},
            {"city": "Baghdad",    "risk_level": "ORANGE", "z_score": 131.7,
             "events": 22, "guide": "물리적 교전 확인. 우선순위 위성 촬영 착수."},
            {"city": "Tel Aviv",   "risk_level": "ORANGE", "z_score": 98.4,
             "events": 17, "guide": "물리적 교전 확인. 우선순위 위성 촬영 착수."},
            {"city": "Beirut",     "risk_level": "YELLOW", "z_score": 45.2,
             "events": 8,  "guide": "긴장 고조. ROI 모니터링 명단 추가."},
            {"city": "Dubai",      "risk_level": "YELLOW", "z_score": 28.3,
             "events": 5,  "guide": "긴장 고조. ROI 모니터링 명단 추가."},
        ]
    }

    risk_cities = {}
    for alert in level1_alerts["alerts"]:
        city = alert["city"]
        risk = alert["risk_level"]
        emoji = {"RED": "🛑", "ORANGE": "🟠", "YELLOW": "🟡"}.get(risk, "⚪")
        risk_cities[city] = risk
        print(f"    {emoji} {risk:8s} | {city:15s} | Z={alert['z_score']:>7.1f} | "
              f"이벤트 {alert['events']:>3d}건 | {alert['guide']}")

    # ──────────────────────────────────────────────
    # STEP 1: TLE 수집
    # ──────────────────────────────────────────────
    print("\n  ── STEP 1: CelesTrak TLE 수집 ──\n")
    tle_data = load_all_tle(force_refresh=False)  # 캐시 있으면 재사용
    print(f"    → {len(tle_data)}개 위성 TLE 확보")
    for norad_id, entry in tle_data.items():
        meta = entry["meta"]
        print(f"       🛰️ {meta['display_name']:12s} (NORAD {norad_id}) | "
              f"{meta['type'].upper():3s} | 해상도 {meta['resolution_m']}m | "
              f"Swath {meta['swath_km']}km")

    # ──────────────────────────────────────────────
    # STEP 2: 궤도 통과 예측 (알람 도시만)
    # ──────────────────────────────────────────────
    print("\n  ── STEP 2: 위성 궤도 통과 예측 (48시간) ──\n")

    # 알람 도시만 추출하여 ROI로 사용
    alert_city_coords = {
        city: ROI_CITIES[city]
        for city in risk_cities.keys()
        if city in ROI_CITIES
    }

    print(f"    → 대상 도시: {', '.join(alert_city_coords.keys())}")
    all_passes = predict_passes(tle_data, cities=alert_city_coords, hours=48)
    swath_passes = filter_shootable(all_passes)
    print(f"    → 전체 통과: {len(all_passes)}건 | Swath 내: {len(swath_passes)}건")

    # ──────────────────────────────────────────────
    # STEP 3: 기상 조건 확인 + 촬영 가능 판정
    # ──────────────────────────────────────────────
    print("\n  ── STEP 3: 기상 조건 확인 ──\n")

    weather_checked = []
    for p in swath_passes:
        checked = check_weather(p)
        checked["risk_level"] = risk_cities.get(p["city"], "N/A")
        checked["priority_score"] = compute_priority_score(checked)
        weather_checked.append(checked)

    shootable = [e for e in weather_checked if e["shootable"]]
    not_shootable = [e for e in weather_checked if not e["shootable"]]

    print(f"    → Swath 내 {len(swath_passes)}건 중:")
    print(f"       ✅ 촬영 가능: {len(shootable)}건")
    print(f"       ❌ 촬영 불가: {len(not_shootable)}건 "
          f"(구름/야간)")

    # ──────────────────────────────────────────────
    # STEP 4: 최종 스케줄 생성 + 우선순위 정렬
    # ──────────────────────────────────────────────
    print("\n  ── STEP 4: 최종 촬영 스케줄 ──\n")

    shootable.sort(key=lambda x: x["priority_score"], reverse=True)

    schedule = {
        "scenario": "Level 1 → Level 2a 통합 시뮬레이션",
        "level1_date": "20260403",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prediction_hours": 48,
        "satellites_tracked": len(tle_data),
        "alert_cities": len(alert_city_coords),
        "total_passes": len(all_passes),
        "swath_passes": len(swath_passes),
        "shootable_passes": len(shootable),
        "level1_alerts": level1_alerts["alerts"],
        "recommendations": shootable,
    }

    # 리포트 출력
    print(f"  {'#':>2s} | {'위성':12s} | {'도시':10s} | {'리스크':8s} | "
          f"{'통과 시각(UTC)':22s} | {'앙각':>5s} | {'구름':>4s} | {'주야':2s} | {'센서':3s} | {'점수':>5s}")
    print("  " + "─" * 95)

    for i, rec in enumerate(shootable, 1):
        daylight_str = "☀️" if rec.get("daylight") else "🌙"
        sensor_str = "EO" if rec["sensor_type"] == "optical" else "SAR"
        risk_emoji = {"RED": "🛑", "ORANGE": "🟠", "YELLOW": "🟡"}.get(rec["risk_level"], "⚪")
        cloud_str = f"{rec.get('cloud_cover_pct', '?')}%"

        print(
            f"  {i:>2d} | {rec['satellite']:12s} | {rec['city']:10s} | "
            f"{risk_emoji} {rec['risk_level']:6s} | {rec['pass_time_utc']:22s} | "
            f"{rec['max_elevation_deg']:5.1f}° | {cloud_str:>4s} | {daylight_str:2s} | "
            f"{sensor_str:3s} | {rec['priority_score']:>5.3f}"
        )

    if not shootable:
        print("  ⚠️ 촬영 가능한 통과 이벤트 없음 — Off-Nadir Phase 2 또는 예측 범위 확대 필요")

    # ──────────────────────────────────────────────
    # STEP 5: JSON 저장
    # ──────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "schedule_20260403_sim.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2, ensure_ascii=False)

    print(f"\n  📄 스케줄 저장: {save_path}")

    # ──────────────────────────────────────────────
    # STEP 6: 운영자 의사결정 지원 요약
    # ──────────────────────────────────────────────
    print("\n" + "═" * 85)
    print("  📋 운영자 의사결정 요약")
    print("═" * 85)

    if shootable:
        best = shootable[0]
        print(f"\n  🥇 최우선 촬영 권고:")
        print(f"     위성: {best['satellite']} ({best['sensor_type'].upper()})")
        print(f"     도시: {best['city']} (리스크: {best['risk_level']})")
        print(f"     시각: {best['pass_time_utc']}")
        print(f"     구름: {best['cloud_cover_pct']}% | 앙각: {best['max_elevation_deg']}°")

        # SAR 백업 탐색
        sar_backup = [r for r in weather_checked if r["sensor_type"] == "sar"]
        if sar_backup:
            sar = sar_backup[0]
            print(f"\n  🔄 SAR 백업 (구름 시):")
            print(f"     위성: {sar['satellite']} | 도시: {sar['city']} | "
                  f"시각: {sar['pass_time_utc']}")
    else:
        print(f"\n  ⚠️ 48시간 내 Swath 범위 촬영 가능 이벤트 없음.")
        print(f"  💡 권고: Off-Nadir 틸팅(Phase 2) 반영 또는 예측 범위를 7일(168시간)로 확대")

    print("\n" + "━" * 85 + "\n")


if __name__ == "__main__":
    main()
