#!/usr/bin/env python3
"""
Run unified Level1+2b -> Level2a pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import PREDICTION_HOURS
from pipeline.integrated_pipeline import run_integrated_pipeline
from pipeline.live_data_preparer import prepare_live_prediction_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="SIA 통합 파이프라인 실행기")
    parser.add_argument("--date", required=True, help="대상 날짜 (YYYYMMDD)")
    parser.add_argument("--hours", type=int, default=PREDICTION_HOURS, help="위성 스케줄 예측 범위(시간)")
    parser.add_argument("--use-llm", action="store_true", help="Level1+2b LLM 검증 활성화")
    parser.add_argument("--top-k", type=int, default=20, help="LLM 검증 및 최종 후보 도시 상한")
    parser.add_argument("--mode", choices=["auto", "operational", "backtest"], default="auto", help="Level2a 실행 모드")
    parser.add_argument("--tle-date", type=str, help="TLE 기준 날짜 (YYYYMMDD)")
    parser.add_argument("--scenario", default="default", help="위성 카탈로그 시나리오")
    parser.add_argument("--refresh", action="store_true", help="TLE 캐시를 무시하고 재수집")
    parser.add_argument("--fetch-gdelt", action="store_true", help="누락된 GDELT 일별 데이터를 받아 merged parquet를 자동 준비")
    parser.add_argument("--main-path", type=str, help="GDELT 메인 parquet 경로 override")
    parser.add_argument("--url-path", type=str, help="GDELT URL parquet 경로 override")
    parser.add_argument("--no-save-level1", action="store_true", help="중간 Level1 JSON 저장 비활성화")
    parser.add_argument("--no-save-schedule", action="store_true", help="최종 Level2a JSON 저장 비활성화")
    parser.add_argument("--verbose", action="store_true", help="전처리/LLM/TLE/PASS/WEATHER 내부 로그 출력")
    args = parser.parse_args()

    resolved_main_path = args.main_path
    resolved_url_path = args.url_path
    resolved_mode = args.mode

    if args.fetch_gdelt:
        prepared = prepare_live_prediction_inputs(
            target_date=args.date,
            main_path=args.main_path,
            url_path=args.url_path,
        )
        resolved_main_path = str(prepared.main_path)
        resolved_url_path = str(prepared.url_path)
        if args.mode == "auto":
            resolved_mode = "operational"
        if prepared.fetched_dates:
            print(
                f"[GDELT] fetched {len(prepared.fetched_dates)} day(s): "
                f"{prepared.fetched_dates[0]} ~ {prepared.fetched_dates[-1]}"
            )
        else:
            print("[GDELT] base parquet already covers target date")

    try:
        result = run_integrated_pipeline(
            target_date=args.date,
            hours=args.hours,
            use_llm=args.use_llm,
            top_k=args.top_k,
            mode=resolved_mode,
            tle_date=args.tle_date,
            refresh=args.refresh,
            scenario=args.scenario,
            save_level1=not args.no_save_level1,
            save_schedule_output=not args.no_save_schedule,
            main_path=resolved_main_path,
            url_path=resolved_url_path,
            verbose=args.verbose,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1) from exc

    if result["schedule"].get("error"):
        print(f"\n[ERROR] {result['schedule']['error']}")


if __name__ == "__main__":
    main()
