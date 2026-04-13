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

from pipeline.level2a import run_level2a_for_date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20260401", help="Level 1 결과를 읽어올 날짜 (YYYYMMDD)")
    parser.add_argument("--hours", type=int, default=72, help="스케줄 예측 범위")
    parser.add_argument("--mode", choices=["operational", "backtest"], default="operational", help="실행 모드")
    parser.add_argument("--tle-date", type=str, help="백테스트용 TLE 기준 날짜 (YYYYMMDD)")
    parser.add_argument("--scenario", default="default", help="위성 카탈로그 시나리오")
    parser.add_argument("--refresh", action="store_true", help="TLE 캐시를 무시하고 메타데이터를 재수집")
    args = parser.parse_args()

    if args.mode == "backtest":
        print(f"\n  ── 과거 TLE 기반 Level 2a 백테스트 가동 ({args.date}) ──\n")
    else:
        print(f"\n  ── 실제 데이터 기반 Level 2a 파이프라인 가동 ({args.date}) ──\n")

    schedule = run_level2a_for_date(
        target_date=args.date,
        hours=args.hours,
        mode=args.mode,
        tle_date=args.tle_date,
        refresh=args.refresh,
        scenario=args.scenario,
        save_output=True,
    )

    if "error" in schedule and schedule.get("missing_path"):
        print(f"  ❌ {args.date}의 Level 1 분석 결과를 찾을 수 없습니다.")
        print(f"     (경로: {schedule['missing_path']})")
        return

    if "error" in schedule and schedule.get("alert_count") == 0:
        print(f"  ✅ {args.date}에는 탐지된 이상 징후가 없습니다. 위성 스케줄링을 생략합니다.")
        return

    if "error" in schedule:
        print("  ❌ 스케줄 생성 실패")
        if schedule.get("mode") == "backtest" and schedule.get("tle_source") == "missing-historical-cache":
            print(
                f"     요청 TLE 기준일: {schedule.get('tle_requested_date')} | "
                "사용 가능한 과거 TLE 캐시 없음"
            )
            print("     data/tle/tle_YYYYMMDD.json 형태의 과거 TLE 스냅샷이 필요합니다.")
        return


if __name__ == "__main__":
    main()
