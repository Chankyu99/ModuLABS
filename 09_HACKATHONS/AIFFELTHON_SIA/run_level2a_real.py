#!/usr/bin/env python3
"""
Compatibility runner for Level2a only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import PREDICTION_HOURS
from pipeline.level2a import run_level2a_for_date


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20260411", help="Level 1 결과를 읽어올 날짜 (YYYYMMDD)")
    parser.add_argument("--hours", type=int, default=PREDICTION_HOURS, help="스케줄 예측 범위")
    parser.add_argument("--mode", choices=["operational", "backtest"], default="operational", help="실행 모드")
    parser.add_argument("--tle-date", type=str, help="백테스트용 TLE 기준 날짜 (YYYYMMDD)")
    parser.add_argument("--scenario", default="default", help="위성 카탈로그 시나리오")
    parser.add_argument("--refresh", action="store_true", help="TLE 캐시를 무시하고 메타데이터를 재수집")
    args = parser.parse_args()

    schedule = run_level2a_for_date(
        target_date=args.date,
        hours=args.hours,
        mode=args.mode,
        tle_date=args.tle_date,
        refresh=args.refresh,
        scenario=args.scenario,
        save_output=True,
    )

    if "error" in schedule:
        print(schedule["error"])


if __name__ == "__main__":
    main()
