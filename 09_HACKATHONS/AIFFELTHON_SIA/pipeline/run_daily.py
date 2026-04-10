#!/usr/bin/env python3
"""
SIA 갈등 모니터링 파이프라인 - 일별 실행 및 백테스트 엔진
──────────────────────────────────────────────────────
사용법:
  python -m pipeline.run_daily                     # 당일 데이터 실행
  python -m pipeline.run_daily --date 20260401     # 특정 날짜 데이터 실행
  python -m pipeline.run_daily --backtest          # 과거 데이터 백테스트 (2026-02-28 ~)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from pipeline.config import OUTPUT_DIR, RISK_LEVELS, LLM_CONFIDENCE_THRESHOLD
from pipeline.conflict_index import compute_conflict_index, detect_anomalies
from pipeline.gdelt_fetcher import load_all_data, fetch_daily
from pipeline.llm_verifier import verify_top_cities, filter_blacklist


def format_report(anomalies: pd.DataFrame, target_date: str) -> str:
    """이상 징후 탐지 결과를 일독용 보고서 형식으로 변환"""
    lines = []
    lines.append(f"\n{'═'*85}")
    lines.append(f"  🛰️  SIA 일일 갈등 모니터링 보고서")
    lines.append(f"  📅 분석 기준일: {target_date}")
    lines.append(f"{'═'*85}")

    today = anomalies[anomalies['date'] == target_date]

    if today.empty:
        lines.append(f"\n  📊 해당 날짜에 모니터링 대상 이벤트가 없습니다.")
        return '\n'.join(lines)

    alerts = today[today['is_anomaly'] == True]
    normals = today[today['is_anomaly'] == False]

    if not alerts.empty:
        lines.append(f"\n  🚨 이상 징후 포착 ({len(alerts)}개 도시)")
        lines.append(f"  {'─'*85}")
        lines.append(f"  {'위험 등급':15s} | {'도시 명칭':15s} | {'갈등(I)':>6s} | {'오차(Z)':>6s} | {'이벤트':>3s} | {'대응 가이드'}")
        lines.append(f"  {'-'*15}-+-{'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*5}-+-{'-'*30}")

        # 위험 등급이 높은 순서로 출력
        for _, row in alerts.sort_values('risk_level', ascending=False).iterrows():
            city_short = row['city'].split(',')[0][:15]
            # LLM 신뢰도 표시
            llm_conf = row.get('llm_confidence', -1)
            if llm_conf >= 0:
                llm_tag = f" [{llm_conf:.0%}]" if llm_conf >= LLM_CONFIDENCE_THRESHOLD else f" ⚠️[{llm_conf:.0%}]"
            else:
                llm_tag = ''
            lines.append(
                f"  {row['risk_emoji']} {row['risk_label']:12s} | {city_short:15s} | "
                f"{row['conflict_index']:>8.0f} | {row['innov_z']:>8.1f} | "
                f"{row['events']:>5.0f} | {row['risk_guide']}{llm_tag}"
            )
    else:
        lines.append(f"\n  ✅ 모든 도시 정상 (이상 징후 없음)")

    if not normals.empty:
        lines.append(f"\n  📊 정기 관측 정보 ({len(normals)}개 도시)")
        # 상위 5개 주요 도시만 요약 출력
        top_normals = normals.nlargest(5, 'conflict_index')
        for _, row in top_normals.iterrows():
            city_short = row['city'].split(',')[0][:20]
            lines.append(
                f"     ⚪ {city_short:20s} | 지수={row['conflict_index']:>6.0f} | "
                f"표준오차={row['innov_z']:>5.1f}"
            )

    lines.append(f"\n{'═'*85}\n")
    return '\n'.join(lines)


def save_result(anomalies: pd.DataFrame, target_date: str):
    """분석 결과를 JSON 형식으로 저장 (대시보드 또는 시스템 연동 목적)"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    today = anomalies[anomalies['date'] == target_date]
    alerts = today[today['is_anomaly'] == True]

    result = {
        'date': target_date,
        'generated_at': datetime.now().isoformat(),
        'total_cities': len(today),
        'alert_count': len(alerts),
        'alerts': [
            {
                'city': r['city'],
                'risk_level': int(r['risk_level']),
                'risk_label': r['risk_label'],
                'conflict_index': round(float(r['conflict_index']), 1),
                'innovation_z': round(float(r['innov_z']), 2),
                'guide': r['risk_guide'],
                'events': int(r['events']),
                'lat': round(float(r['lat']), 4) if pd.notna(r.get('lat')) else None,
                'lon': round(float(r['lon']), 4) if pd.notna(r.get('lon')) else None,
                'country_code': r.get('country_code', ''),
                'llm_confidence': round(float(r.get('llm_confidence', -1)), 2),
                'llm_reason': r.get('llm_reason', ''),
            } for _, r in alerts.iterrows()
        ],
        'summary': [
            {
                'city': r['city'],
                'risk_level': int(r['risk_level']),
                'conflict_index': round(float(r['conflict_index']), 1),
                'z_score': round(float(r['innov_z']), 2),
                'lat': round(float(r['lat']), 4) if pd.notna(r.get('lat')) else None,
                'lon': round(float(r['lon']), 4) if pd.notna(r.get('lon')) else None,
            } for _, r in today.nlargest(10, 'conflict_index').iterrows()
        ]
    }

    save_path = OUTPUT_DIR / f"{target_date}.json"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  [결과 저장] {save_path.name}")
    return save_path


def run_single_day(target_date: str, fetch: bool = False, use_llm: bool = True):
    """특정 날짜의 파이프라인 전체 실행"""
    print(f"\n--- 파이프라인 가동: {target_date} ---")

    if fetch: fetch_daily(target_date)

    # 데이터 로드 (기존 대용량 Parquet + 수집된 일별 Parquet)
    raw = load_all_data(target_date=target_date)
    if raw.empty:
        print("  [오류] 분석 가능한 데이터가 없습니다.")
        return

    # 1. 갈등 지수 산출
    city_daily = compute_conflict_index(raw)
    
    # 2. 칼만 필터 기반 이상 징후 탐지
    results = detect_anomalies(city_daily)

    # 3. LLM 게이트키퍼 검증
    if use_llm:
        import pandas as pd
        url_path = Path(__file__).resolve().parent.parent / 'gdelt_url_final.parquet'
        if url_path.exists():
            url_df = pd.read_parquet(url_path, columns=['GLOBALEVENTID', 'SOURCEURL'])
            results = verify_top_cities(results, raw, url_df, target_date)
        else:
            print("  [LLM] URL 매핑 파일 없음. 블랙리스트만 적용.")
            results = filter_blacklist(results)
    else:
        results = filter_blacklist(results)

    # 4. 리포트 생성 및 저장
    report = format_report(results, target_date)
    print(report)
    save_result(results, target_date)

    return results


def run_backtest():
    """과거 이란 전쟁 기간의 소급 분석(Backtest) 수행"""
    print("\n" + "=" * 85)
    print(f"  🛰️  SIA 분쟁 탐지 모델 소급 검증(Backtest): 2026-02-25 ~ 2026-03-24")
    print("=" * 85)

    raw = load_all_data(include_daily=False)
    if raw.empty: return

    city_daily = compute_conflict_index(raw)
    results = detect_anomalies(city_daily)

    # 검증 대상 기간 필터링
    test_dates = sorted(results[
        (results['date'] >= '20260225') & (results['date'] <= '20260324')
    ]['date'].unique())

    print(f"\n  {'분석 기준일':>10s} | {'경보 건수':>8s} | 주요 포착 도시 (Z-Score)")
    print(f"  {'-'*10}-+-{'-'*8}-+{'-'*60}")

    for date in test_dates:
        day_alerts = results[(results['date'] == date) & (results['is_anomaly'] == True)]
        count = len(day_alerts)

        if count > 0:
            # 상위 5개 도시만 한 줄에 요약
            cities = ", ".join([
                f"{r['city'].split(',')[0][:12]}(Z={r['innov_z']:.1f})"
                for _, r in day_alerts.sort_values('innov_z', ascending=False).head(5).iterrows()
            ])
            print(f"  {date:>10s} | {count:>6} alert | {cities}")

    return results


def main():
    parser = argparse.ArgumentParser(description='SIA 갈등 감시 파이프라인')
    parser.add_argument('--date', type=str, default=None, help='대상 날짜 (YYYYMMDD)')
    parser.add_argument('--fetch', action='store_true', help='GDELT에서 데이터 직접 수집')
    parser.add_argument('--backtest', action='store_true', help='과거 데이터 백테스트 실행')
    parser.add_argument('--no-llm', action='store_true', help='LLM 검증 비활성화')
    args = parser.parse_args()

    if args.backtest:
        run_backtest()
    else:
        target = args.date or datetime.now().strftime('%Y%m%d')
        run_single_day(target, fetch=args.fetch, use_llm=not args.no_llm)


if __name__ == '__main__':
    main()
