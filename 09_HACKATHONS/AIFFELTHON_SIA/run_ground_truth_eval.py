#!/usr/bin/env python3
"""
Ground Truth 전체 날짜 일괄 평가 스크립트
─────────────────────────────────────────
data/ground_truth/ 아래 CSV 파일명에서 날짜를 추출하여
run_daily.py 파이프라인을 순서대로 실행합니다.

사용법:
  python run_ground_truth_eval.py              # LLM 없이 실행
  python run_ground_truth_eval.py --use-llm   # LLM 검증 포함
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (pipeline 패키지 임포트용)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.run_daily import run_single_day

GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "ground_truth"
OUTPUT_DIR       = PROJECT_ROOT / "output"


def save_gemma_input(dates: list[str]) -> Path:
    """모든 날짜의 모델 출력 + Ground Truth를 Gemma 분석용 단일 텍스트 파일로 저장."""
    lines = []
    lines.append("# SIA 백테스트 분석 데이터")
    lines.append(f"# 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"# 분석 대상: {len(dates)}개 날짜  ({', '.join(dates)})")
    lines.append("")
    lines.append("## 분석 요청")
    lines.append("아래 각 날짜별 [모델 예측]과 [Ground Truth]를 비교하여 다음을 한국어로 분석해주세요:")
    lines.append("1. True Positive  : 모델이 경보를 냈고 GT에도 있는 도시")
    lines.append("2. False Negative : GT에 있는데 모델이 놓친 도시 (탐지 실패)")
    lines.append("3. False Positive : 모델이 경보를 냈는데 GT에 없는 도시 (오탐)")
    lines.append("4. Precision : 모델이 경보를 낸 것 중 실제 갈등 상황인 도시의 비율")
    lines.append("5. Recall : 모델이 실제 갈등 상황을 얼마나 잘 탐지했는지의 비율")
    lines.append("6. 날짜별 패턴과 모델의 전반적인 성능 및 개선 방향")
    lines.append("")

    for date in dates:
        lines.append(f"{'─'*70}")
        lines.append(f"## 날짜: {date}")
        lines.append("")

        # ── 모델 예측 결과 ──
        lines.append("### [모델 예측]")
        output_path = OUTPUT_DIR / f"{date}.json"
        if output_path.exists():
            with open(output_path, encoding="utf-8") as f:
                data = json.load(f)

            alerts = data.get("alerts", [])
            if alerts:
                lines.append(f"정밀 촬영 후보 ({len(alerts)}개):")
                for a in alerts:
                    lines.append(
                        f"  - {a['city']} | 갈등지수={a['conflict_index']} | Z={a['innovation_z']} "
                        f"| {a.get('llm_event_summary', '').strip()}"
                    )
            else:
                lines.append("  정밀 촬영 후보 없음")

            suppressed = data.get("suppressed_count", 0)
            summary = data.get("summary", [])
            if suppressed > 0:
                lines.append(f"분석관 판단 필요 도시: {suppressed}개")

            if summary:
                lines.append(f"전체 상위 도시 (갈등지수 기준 Top {len(summary)}):")
                for s in summary:
                    lines.append(
                        f"  - {s['city']} | 갈등지수={s['conflict_index']} | Z={s['z_score']} "
                        f"| 검증={s.get('llm_validation_type', 'N/A')}"
                    )
        else:
            lines.append("  (출력 파일 없음 — 해당 날짜 실행 실패)")
        lines.append("")

        # ── Ground Truth ──
        lines.append("### [Ground Truth]")
        gt_path = GROUND_TRUTH_DIR / f"{date}.csv"
        if gt_path.exists():
            with open(gt_path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lines.append(
                        f"  - {row['ActionGeo_FullName']} | {row.get('Event_Description', '').strip()}"
                    )
        else:
            lines.append("  (Ground Truth 파일 없음)")
        lines.append("")

    lines.append(f"{'─'*70}")
    lines.append("## 종합 평가를 위 데이터를 바탕으로 작성해주세요.")

    save_path = OUTPUT_DIR / "backtest_gemma_input.txt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines), encoding="utf-8")
    return save_path


def get_gt_dates() -> list[str]:
    """ground_truth 디렉토리에서 날짜 목록을 추출합니다."""
    dates = sorted([
        p.stem for p in GROUND_TRUTH_DIR.glob("*.csv")
        if p.stem.isdigit() and len(p.stem) == 8
    ])
    return dates


def main():
    parser = argparse.ArgumentParser(description="Ground Truth 전체 날짜 일괄 평가")
    parser.add_argument("--use-llm", action="store_true", help="LLM 검증 활성화")
    args = parser.parse_args()

    dates = get_gt_dates()
    if not dates:
        print(f"[오류] {GROUND_TRUTH_DIR} 에서 날짜 파일을 찾을 수 없습니다.")
        sys.exit(1)

    total = len(dates)
    print(f"\n{'═'*70}")
    print(f"  SIA Ground Truth 일괄 평가")
    print(f"  대상 날짜: {total}개  |  LLM: {'ON' if args.use_llm else 'OFF'}")
    print(f"  {' / '.join(dates)}")
    print(f"{'═'*70}\n")

    results_summary = []

    COOLDOWN_SEC = 120  # 날짜 간 API 과부하 방지 대기 시간

    for i, date in enumerate(dates, 1):
        print(f"\n[{i}/{total}] ▶ {date} 처리 시작")
        t0 = time.time()
        try:
            run_single_day(date, fetch=False, use_llm=args.use_llm)
            elapsed = time.time() - t0
            results_summary.append((date, "✅ 완료", f"{elapsed:.1f}s"))
            print(f"[{i}/{total}] ✅ {date} 완료 ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            results_summary.append((date, "❌ 실패", str(e)))
            print(f"[{i}/{total}] ❌ {date} 실패: {e}")

        if args.use_llm and i < total:
            print(f"  ⏳ API 과부하 방지 대기 중... ({COOLDOWN_SEC}s)")
            time.sleep(COOLDOWN_SEC)

    print(f"\n{'═'*70}")
    print(f"  최종 결과 요약")
    print(f"{'─'*70}")
    for date, status, detail in results_summary:
        print(f"  {date}  {status}  {detail}")
    print(f"{'═'*70}\n")

    gemma_file = save_gemma_input(dates)
    print(f"  📄 Gemma 분석용 파일 저장 완료: {gemma_file.name}\n")


if __name__ == "__main__":
    main()
