#!/usr/bin/env python3
"""
일별 실행 스크립트
──────────────────
사용법:
  python -m pipeline.run_daily                     # 오늘 날짜 기준
  python -m pipeline.run_daily --date 20260401     # 특정 날짜
  python -m pipeline.run_daily --backtest          # 2026년 2~3월 백테스트
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from pipeline.config import OUTPUT_DIR, Z_THRESHOLD
from pipeline.conflict_index import compute_conflict_index, detect_anomalies
from pipeline.gdelt_fetcher import load_all_data, fetch_daily


def format_report(anomalies: pd.DataFrame, target_date: str) -> str:
    """이상 탐지 결과를 사람이 읽기 좋은 보고서로 포맷"""
    lines = []
    lines.append(f"\n{'═'*70}")
    lines.append(f"  🛰️  SIA 일일 분쟁 모니터링 보고서")
    lines.append(f"  📅 {target_date}")
    lines.append(f"{'═'*70}")

    today = anomalies[anomalies['date'] == target_date]

    if today.empty:
        lines.append(f"\n  📊 해당 날짜에 Triad 이벤트 없음")
        return '\n'.join(lines)

    alerts = today[today['is_anomaly'] == True]
    normals = today[today['is_anomaly'] == False]

    if not alerts.empty:
        lines.append(f"\n  🚨 이상 징후 도시 ({len(alerts)}개)")
        lines.append(f"  {'─'*65}")
        lines.append(f"  {'도시':20s} | {'I':>8s} | {'예측':>8s} | {'Innovation':>10s} | {'Z':>6s} | {'건수':>4s} | {'Tone':>5s}")
        lines.append(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}-+-{'-'*4}-+-{'-'*5}")

        for _, row in alerts.iterrows():
            city_short = row['city'].split(',')[0][:20]
            lines.append(
                f"  {city_short:20s} | {row['conflict_index']:>8.0f} | "
                f"{row['kalman_est']:>8.0f} | {row['innovation']:>10.0f} | "
                f"{row['innov_z']:>5.1f} | {row['events']:>4.0f} | "
                f"{row['avg_tone']:>5.1f}"
            )
    else:
        lines.append(f"\n  ✅ 이상 징후 없음")

    if not normals.empty:
        lines.append(f"\n  📊 정상 모니터링 중 ({len(normals)}개 도시)")
        top_normals = normals.nlargest(5, 'conflict_index')
        for _, row in top_normals.iterrows():
            city_short = row['city'].split(',')[0][:20]
            lines.append(
                f"     ⚪ {city_short:20s} | I={row['conflict_index']:>6.0f} | "
                f"Z={row['innov_z']:>5.1f} | {row['events']:.0f}건"
            )

    lines.append(f"\n{'═'*70}\n")
    return '\n'.join(lines)


def save_result(anomalies: pd.DataFrame, target_date: str):
    """결과를 JSON으로 저장"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    today = anomalies[anomalies['date'] == target_date]
    alerts = today[today['is_anomaly'] == True]

    result = {
        'date': target_date,
        'generated_at': datetime.now().isoformat(),
        'total_cities': len(today),
        'alert_count': len(alerts),
        'alerts': [],
        'summary': [],
    }

    for _, row in alerts.iterrows():
        result['alerts'].append({
            'city': row['city'],
            'conflict_index': round(float(row['conflict_index']), 1),
            'kalman_estimate': round(float(row['kalman_est']), 1),
            'innovation': round(float(row['innovation']), 1),
            'innovation_z': round(float(row['innov_z']), 2),
            'events': int(row['events']),
            'mentions': int(row['mentions']),
            'avg_tone': round(float(row['avg_tone']), 2),
        })

    for _, row in today.nlargest(10, 'conflict_index').iterrows():
        result['summary'].append({
            'city': row['city'],
            'conflict_index': round(float(row['conflict_index']), 1),
            'z_score': round(float(row['innov_z']), 2),
            'is_anomaly': bool(row['is_anomaly']),
        })

    save_path = OUTPUT_DIR / f"{target_date}.json"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  💾 결과 저장: {save_path.name}")
    return save_path


def run_single_day(target_date: str, fetch: bool = False):
    """특정 날짜에 대해 파이프라인 실행"""
    print(f"\n  ⏳ 파이프라인 실행: {target_date}")

    # 1. 데이터 수집 (fetch=True면 GDELT에서 다운로드)
    if fetch:
        fetch_daily(target_date)

    # 2. 전체 데이터 로드
    raw = load_all_data()
    if raw.empty:
        print("  ❌ 데이터 없음")
        return

    # 3. 갈등지수 산출
    print(f"  🔧 갈등지수 산출 중...")
    city_daily = compute_conflict_index(raw)
    print(f"     {len(city_daily):,}건 (도시-일자 조합)")

    # 4. 이상탐지
    print(f"  🔍 칼만 필터 + Innovation Z-score 계산 중...")
    results = detect_anomalies(city_daily, threshold=Z_THRESHOLD)

    # 5. 보고서 출력
    report = format_report(results, target_date)
    print(report)

    # 6. 결과 저장
    save_result(results, target_date)

    return results


def run_backtest():
    """2026년 2~3월 데이터로 백테스트"""
    print("\n" + "=" * 70)
    print("  🔬 백테스트 모드: 2026년 2월 25일 ~ 3월 24일")
    print("=" * 70)

    # 기존 데이터만 사용 (다운로드 안 함)
    raw = load_all_data(include_daily=False)
    if raw.empty:
        print("  ❌ 데이터 없음")
        return

    city_daily = compute_conflict_index(raw)
    results = detect_anomalies(city_daily, threshold=Z_THRESHOLD)

    # 날짜별 경보 요약
    test_dates = sorted(results[
        (results['date'] >= '20260225') & (results['date'] <= '20260324')
    ]['date'].unique())

    print(f"\n  {'날짜':>10s} | {'경보 도시':>8s} | 상세")
    print(f"  {'-'*10}-+-{'-'*8}-+{'-'*50}")

    total_alerts = 0
    for date in test_dates:
        day_alerts = results[
            (results['date'] == date) & (results['is_anomaly'] == True)
        ]
        count = len(day_alerts)
        total_alerts += count

        if count > 0:
            cities = ", ".join([
                f"{r['city'].split(',')[0][:12]}(Z={r['innov_z']:.1f})"
                for _, r in day_alerts.iterrows()
            ])
            print(f"  {date:>10s} | {count:>6}개 | {cities}")

    alert_days = len([d for d in test_dates
                      if len(results[(results['date'] == d) & results['is_anomaly']]) > 0])

    print(f"\n  📊 백테스트 요약:")
    print(f"     기간: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}일)")
    print(f"     경보 발생일: {alert_days}일")
    print(f"     총 경보 건수: {total_alerts}건")
    print(f"     일평균 경보: {total_alerts/len(test_dates):.1f}건")


def main():
    parser = argparse.ArgumentParser(description='SIA 분쟁 모니터링 파이프라인')
    parser.add_argument('--date', type=str, default=None,
                       help='실행 날짜 (YYYYMMDD). 기본값: 오늘')
    parser.add_argument('--fetch', action='store_true',
                       help='GDELT에서 데이터 다운로드')
    parser.add_argument('--backtest', action='store_true',
                       help='2026년 2~3월 백테스트')

    args = parser.parse_args()

    if args.backtest:
        run_backtest()
    else:
        target = args.date or datetime.now().strftime('%Y%m%d')
        run_single_day(target, fetch=args.fetch)


if __name__ == '__main__':
    main()
