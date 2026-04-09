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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20260401", help="Level 1 결과를 읽어올 날짜 (YYYYMMDD)")
    parser.add_argument("--hours", type=int, default=48, help="스케줄 예측 범위")
    args = parser.parse_args()

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

    # 2. 알람 도시 추출 (ROI 필터링)
    risk_cities = {}
    valid_cities = {}
    
    print("  [Level 1] 다음 도시에 대한 위기 경보 접수:")
    for a in alerts:
        city_full = a['city']
        city = city_full.split(',')[0].strip()  # "Tel Aviv,Israel" -> "Tel Aviv"
        risk_label = a['risk_label'].split('(')[-1].strip(')') # "RED", "ORANGE" 등 추출
        
        # config.py의 ROI_CITIES에 좌표 데이터가 있는 도시만 선별
        if city in ROI_CITIES:
            risk_cities[city] = risk_label
            valid_cities[city] = ROI_CITIES[city]
            
            emoji = {"RED": "🛑", "ORANGE": "🟠", "YELLOW": "🟡"}.get(risk_label, "⚪")
            print(f"    {emoji} {city:15s} ({risk_label:6s}) - Z-score: {a['innovation_z']}")
        else:
            print(f"    ⚠️ {city} 좌표 미등록 (ROI_CITIES 제외됨)")

    if not valid_cities:
        print("\n  ❌ 스케줄링 가능한 ROI 대상 도시가 없습니다.")
        return

    # 3. Schedule Builder 가동
    # build_schedule: (cities, risk_cities, hours)
    schedule = build_schedule(
        cities=valid_cities,
        risk_cities=risk_cities,
        hours=args.hours,
        force_tle_refresh=False
    )

    if "error" in schedule:
        print("  ❌ 스케줄 생성 실패")
        return

    # 4. 결과 출력 및 저장
    print_schedule(schedule)
    save_path = save_schedule(schedule, filename=f"schedule_{args.date}_real.json")


if __name__ == "__main__":
    main()
