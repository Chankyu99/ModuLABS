#!/usr/bin/env python3
"""
Level 1 결과 연동 실제 Level 2a 실행
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Level 1 파이프라인의 아웃풋 JSON({date}.json)을 파싱하여,
알람이 발생한 도시를 대상으로 위성 촬영 스케줄을 생성합니다.
"""
from __future__ import annotations

import json
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

# 파이프라인 루트 설정
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import ROI_CITIES, OUTPUT_DIR
from pipeline.schedule_builder import build_schedule, save_schedule, print_schedule


def load_level1_results(date_str: str) -> dict | None:
    """해당 날짜의 Level 1 output JSON을 로드합니다."""
    json_path = OUTPUT_DIR / f"{date_str}.json"
    if not json_path.exists():
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_city_coordinates(city: str, alert: dict) -> tuple[dict | None, str]:
    """Level 1 결과 좌표를 우선 사용하고, 없으면 정적 ROI를 fallback으로 쓴다."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20260401", help="Level 1 결과를 읽어올 날짜 (YYYYMMDD)")
    parser.add_argument("--hours", type=int, default=72, help="스케줄 예측 범위")
    parser.add_argument("--mode", choices=["operational", "backtest"], default="operational", help="실행 모드")
    parser.add_argument("--tle-date", type=str, help="백테스트용 TLE 기준 날짜 (YYYYMMDD)")
    parser.add_argument("--refresh", action="store_true", help="TLE 캐시를 무시하고 메타데이터를 재수집")
    args = parser.parse_args()

    if args.mode == "backtest":
        print(f"\n  ── 과거 TLE 기반 Level 2a 백테스트 가동 ({args.date}) ──\n")
    else:
        print(f"\n  ── 실제 데이터 기반 Level 2a 파이프라인 가동 ({args.date}) ──\n")

    # 1. Level 1 데이터 로드
    l1_data = load_level1_results(args.date)
    if not l1_data:
        print(f"  ❌ {args.date}의 Level 1 분석 결과를 찾을 수 없습니다.")
        print(f"     (경로: {OUTPUT_DIR / f'{args.date}.json'})")
        return

    alerts = l1_data.get("alerts", [])
    if not alerts:
        print(f"  ✅ {args.date}에는 탐지된 이상 징후가 없습니다. 위성 스케줄링을 생략합니다.")
        return

    # 2. 알람 도시 추출 (Level 1 좌표 우선, ROI fallback)
    risk_cities = {}
    valid_cities = {}
    
    print("  [Level 1] 다음 도시에 대한 위기 경보 접수:")
    for a in alerts:
        city_full = a['city']
        city = city_full.split(',')[0].strip()  # "Tel Aviv,Israel" -> "Tel Aviv"
        risk_label = a['risk_label'].split('(')[-1].strip(')') # "RED", "ORANGE" 등 추출
        
        coords, coord_source = resolve_city_coordinates(city, a)

        if coords is not None:
            risk_cities[city] = {
                "risk_label": risk_label,
                "innovation_z": a.get("innovation_z", 0.0),
                "severity_score": a.get("innovation_z", 0.0),
                "lat": coords["lat"],
                "lon": coords["lon"],
                "country_code": a.get("country_code", ""),
            }
            valid_cities[city] = coords
            
            emoji = {"RED": "🛑", "ORANGE": "🟠", "YELLOW": "🟡"}.get(risk_label, "⚪")
            source_tag = "L1" if coord_source == "level1" else "ROI"
            print(
                f"    {emoji} {city:15s} ({risk_label:6s}) - Z-score: {a['innovation_z']} "
                f"[coord:{source_tag}]"
            )
        else:
            print(f"    ⚠️ {city} 좌표 미확보 (Level 1/ROI 모두 없음)")

    if not valid_cities:
        print("\n  ❌ 스케줄링 가능한 ROI 대상 도시가 없습니다.")
        return

    if args.mode == "backtest":
        prediction_start_utc = datetime.strptime(args.date, "%Y%m%d").replace(tzinfo=timezone.utc)
        tle_reference_date = args.tle_date or args.date
    else:
        prediction_start_utc = datetime.now(timezone.utc)
        tle_reference_date = args.tle_date or prediction_start_utc.strftime("%Y%m%d")

    # 3. Schedule Builder 가동
    # build_schedule: (cities, risk_cities, hours)
    schedule = build_schedule(
        cities=valid_cities,
        risk_cities=risk_cities,
        hours=args.hours,
        force_tle_refresh=args.refresh,
        tle_mode=args.mode,
        tle_reference_date=tle_reference_date,
        prediction_start_utc=prediction_start_utc,
    )

    if "error" in schedule:
        print("  ❌ 스케줄 생성 실패")
        if schedule.get("mode") == "backtest" and schedule.get("tle_source") == "missing-historical-cache":
            print(
                f"     요청 TLE 기준일: {schedule.get('tle_requested_date')} | "
                "사용 가능한 과거 TLE 캐시 없음"
            )
            print("     data/tle/tle_YYYYMMDD.json 형태의 과거 TLE 스냅샷이 필요합니다.")
        return

    # 4. 결과 출력 및 저장
    print_schedule(schedule)
    suffix = "backtest" if args.mode == "backtest" else "real"
    save_schedule(schedule, filename=f"schedule_{args.date}_{suffix}.json")


if __name__ == "__main__":
    main()
