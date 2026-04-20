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

from pipeline.city_utils import normalize_city_key
from pipeline.config import (
    OUTPUT_DIR,
    RISK_LEVELS,
    LLM_ALLOW_STRATEGIC_SINGLE_SUPPORT,
    LLM_ALLOW_ROI_PRIOR_SINGLE_SUPPORT,
    CITY_BLACKLIST,
    LLM_MIN_EXACT_SUPPORT,
    LLM_ALLOW_SINGLE_ARTICLE_EXACT,
    REPORT_MAX_ALERTS,
    REPORT_MIN_CONFLICT_INDEX,
    REPORT_MIN_INNOV_Z,
)
from pipeline.conflict_index import compute_conflict_index, detect_anomalies
from pipeline.gdelt_fetcher import load_all_data, fetch_daily


def filter_blacklist(results: pd.DataFrame) -> pd.DataFrame:
    """LLM 없이도 적용 가능한 최소 도시명 블랙리스트 필터."""
    if results.empty or 'city' not in results.columns:
        return results

    blocked = {name.lower() for name in CITY_BLACKLIST}
    city_lower = results['city'].fillna('').astype(str).str.lower()
    keep_mask = ~city_lower.isin(blocked)
    return results.loc[keep_mask].copy()


def attach_report_urls(
    results: pd.DataFrame,
    raw_df: pd.DataFrame,
    url_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """도시/날짜별 대표 기사 URL을 결과에 붙여 리포트 출력에 활용한다."""
    if results.empty:
        enriched = results.copy()
        if 'report_source_url' not in enriched.columns:
            enriched['report_source_url'] = pd.Series(dtype='object')
        return enriched

    enriched = results.copy()
    if 'report_source_url' not in enriched.columns:
        enriched['report_source_url'] = ''
    enriched['report_source_url'] = enriched['report_source_url'].fillna('').astype(str)

    if raw_df.empty or 'SQLDATE' not in raw_df.columns or 'ActionGeo_FullName' not in raw_df.columns:
        return enriched

    raw = raw_df.copy()
    raw['date'] = raw['SQLDATE'].astype(str).str[:8]
    raw['city_key'] = raw['ActionGeo_FullName'].astype(str).map(normalize_city_key)

    if 'SOURCEURL' not in raw.columns:
        raw['SOURCEURL'] = None

    if url_df is not None and not url_df.empty and 'GLOBALEVENTID' in raw.columns:
        mapping = url_df.copy()
        mapping['GLOBALEVENTID'] = mapping['GLOBALEVENTID'].astype(str)
        raw['GLOBALEVENTID'] = raw['GLOBALEVENTID'].astype(str)
        raw = raw.merge(
            mapping[['GLOBALEVENTID', 'SOURCEURL']].rename(columns={'SOURCEURL': 'MAPPED_SOURCEURL'}),
            on='GLOBALEVENTID',
            how='left',
        )
        raw['SOURCEURL'] = raw['SOURCEURL'].fillna(raw['MAPPED_SOURCEURL'])
        raw = raw.drop(columns=['MAPPED_SOURCEURL'])

    for col in ['NumSources', 'NumMentions', 'NumArticles']:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors='coerce').fillna(0.0)
        else:
            raw[col] = 0.0

    raw['candidate_weight'] = raw['NumSources'] * 4 + raw['NumMentions'] + raw['NumArticles']
    raw['SOURCEURL'] = raw['SOURCEURL'].fillna('').astype(str).str.strip()

    candidates = raw.loc[
        raw['SOURCEURL'].str.startswith(('http://', 'https://'))
        & raw['city_key'].ne('')
    ].copy()
    if candidates.empty:
        return enriched

    top_urls = (
        candidates
        .sort_values(
            ['date', 'city_key', 'candidate_weight', 'NumSources', 'NumMentions', 'NumArticles'],
            ascending=[True, True, False, False, False, False],
        )
        .drop_duplicates(subset=['date', 'city_key'], keep='first')
        [['date', 'city_key', 'SOURCEURL']]
        .rename(columns={'SOURCEURL': 'report_source_url_selected'})
    )

    enriched['city_key'] = enriched['city'].astype(str).map(normalize_city_key)
    enriched = enriched.merge(top_urls, on=['date', 'city_key'], how='left')
    selected_urls = enriched['report_source_url_selected'].fillna('').astype(str)
    enriched['report_source_url'] = enriched['report_source_url'].where(
        enriched['report_source_url'].str.strip().ne(''),
        selected_urls,
    )

    return enriched.drop(columns=['city_key', 'report_source_url_selected'])


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

    alerts = today[today['is_anomaly'] == True].copy()
    normals = today[today['is_anomaly'] == False]

    def safe_text(value, default: str = "") -> str:
        if pd.isna(value):
            return default
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return default
        return text

    def build_core_message(row: pd.Series) -> str:
        summary = safe_text(row.get('llm_event_summary', ''))
        imagery = safe_text(row.get('llm_imagery_need', ''))
        if summary and imagery:
            return f"{summary} {imagery}"
        if summary:
            return summary
        if imagery:
            return imagery
        return safe_text(row.get('risk_guide', ''), "사건 핵심 요약을 추가 검토해야 합니다.")

    default_actionability = 'suppress' if 'llm_actionability' in alerts.columns else 'actionable'
    alerts['llm_actionability'] = alerts.get(
        'llm_actionability',
        pd.Series(default_actionability, index=alerts.index),
    ).fillna(default_actionability).astype(str)

    score_mask = (
        (pd.to_numeric(alerts['conflict_index'], errors='coerce').fillna(0.0) >= REPORT_MIN_CONFLICT_INDEX)
        & (pd.to_numeric(alerts['innov_z'], errors='coerce').fillna(0.0) >= REPORT_MIN_INNOV_Z)
    )
    actionable_alerts = (
        alerts.loc[score_mask & alerts['llm_actionability'].eq('actionable')]
        .sort_values(['innov_z', 'conflict_index'], ascending=[False, False])
        .head(REPORT_MAX_ALERTS)
        .copy()
    )
    suppressed_count = int((score_mask & ~alerts['llm_actionability'].eq('actionable')).sum())

    lines.append(f"\n  🎯 [정밀 촬영 후보 선정 기준]")
    lines.append(f"     1. 데이터 급증: 갈등 지수가 {REPORT_MIN_CONFLICT_INDEX:.0f} 이상이며, 평소 대비 보도량이 폭증(Z-score {REPORT_MIN_INNOV_Z:.1f} 이상)한 지역")
    lines.append(f"     2. AI 교차 검증: 실제 물리적 타격이나 군사적 피해가 발생했다고 AI가 2건 이상의 기사로 팩트체크한 지역")
    if LLM_ALLOW_SINGLE_ARTICLE_EXACT:
        lines.append(f"        (※ 단, 초기 속보이거나 핵심 전략 표적(공항/항만/핵·군사시설)인 경우 1건의 확실한 보도만 있어도 긴급 승격)")

    if not actionable_alerts.empty:
        lines.append(f"\n  🚨 정밀 촬영 후보 ({len(actionable_alerts)}개 도시)")
        lines.append(f"  {'─'*85}")
        lines.append(f"  {'도시 명칭':15s} | {'갈등(I)':>8s} | {'오차(Z)':>8s} | {'이벤트':>5s} | {'검증'}")
        lines.append(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}-+-{'-'*20}")

        for _, row in actionable_alerts.iterrows():
            city_short = row['city'].split(',')[0][:15]
            llm_conf = row.get('llm_confidence', -1)
            report_url = safe_text(row.get('report_source_url', ''))
            resolved_city = safe_text(row.get('llm_resolved_city', ''))
            validation_type = safe_text(row.get('llm_validation_type', ''), 'validated')
            exact_support = int(row.get('llm_exact_support', 0) or 0)
            nearby_support = int(row.get('llm_nearby_support', 0) or 0)
            strategic_support = int(row.get('llm_strategic_support', 0) or 0)
            roi_prior_support = int(row.get('llm_roi_prior_support', 0) or 0)
            article_count = int(row.get('llm_article_count', 0) or 0)
            target_category = safe_text(row.get('llm_target_category', ''))
            if not pd.isna(llm_conf) and float(llm_conf) >= 0:
                if resolved_city and resolved_city != str(row['city']).strip():
                    llm_tag = f" [{validation_type}:{resolved_city}|exact={exact_support}/{article_count}|strategic={strategic_support}|roi={roi_prior_support}]"
                else:
                    llm_tag = f" [{validation_type}|exact={exact_support}/{article_count}|nearby={nearby_support}|strategic={strategic_support}|roi={roi_prior_support}]"
            else:
                llm_tag = '-'
            lines.append(
                f"  {city_short:15s} | "
                f"{row['conflict_index']:>8.0f} | {row['innov_z']:>8.1f} | "
                f"{row['events']:>5.0f} | {llm_tag}"
            )
            if target_category:
                lines.append(f"     표적: {target_category}")
            lines.append(f"     핵심: {build_core_message(row)}")
            if report_url:
                lines.append(f"     URL: {report_url}")
    else:
        lines.append(f"\n  ✅ 정밀 촬영 후보 없음 (현재 조건 기준)")

    # 이 아래 부분은 출력 결과에 없어도 될 것 같음 -> 추후 삭제 필요
    if suppressed_count > 0:
        lines.append(f"\n ⚙️ LLM이 근거 부족/모호성으로 숨긴 후보: {suppressed_count}개")
    
        suppressed_alerts = alerts.loc[score_mask & ~alerts['llm_actionability'].eq('actionable')].sort_values('innov_z', ascending=False)
        for _, row in suppressed_alerts.iterrows():
            city_name = str(row['city']).split(',')[0][:15]
            z_score = float(row['innov_z'])
            reason = str(row.get('llm_reason', '사유 없음')).replace('\n', ' ')[:60] # 사유가 길면 60자에서 자름
            val_type = str(row.get('llm_validation_type', ''))
                
            lines.append(f"     - {city_name:15s} | Z={z_score:>5.1f} | [{val_type}] {reason}...")


    if not normals.empty:
        lines.append(f"\n  📊 [TRACK 2] 정기 전략 관측 및 전황 브리핑 (상시 집중 모니터링)")
        lines.append(f"  {'─'*85}")
        lines.append(f"  전쟁 양상 파악을 위해 꾸준한 위성 자원 할당이 필요한 핵심 거점입니다.")
        lines.append(f"  (선정 기준: 갈등 지수 5,000 이상 핵심 전략지)")
        
        # Track 2 조건 (Z-score는 낮지만, 볼륨이 5000 이상인 곳)
        track2_mask = normals['conflict_index'] >= 5000
        top_normals = normals[track2_mask].sort_values('conflict_index', ascending=False).head(5)
        
        if top_normals.empty:
             lines.append(f"\n     ✅ 현재 기준을 충족하는 대형 정기 관측 도시가 없습니다.")
        else:
            for _, row in top_normals.iterrows():
                city_name = str(row['city']).split(',')[0]
                summary = str(row.get('llm_baseline_summary', '요약 정보 없음'))
                report_url = safe_text(row.get('report_source_url', ''))
                lines.append(
                    f"\n     📍 {city_name:15s} | 지수={row['conflict_index']:>6.0f} | 표준오차={row['innov_z']:>5.1f}"
                )
                lines.append(f"     📌 전황: {summary}")
                if report_url:
                    lines.append(f"     🔗 기사: {report_url}")

    lines.append(f"\n{'═'*85}\n")
    return '\n'.join(lines)


def save_result(anomalies: pd.DataFrame, target_date: str):
    """분석 결과를 JSON 형식으로 저장 (대시보드 또는 시스템 연동 목적)"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    today = anomalies[anomalies['date'] == target_date].copy()
    alerts = today[today['is_anomaly'] == True].copy()

    def safe_json_text(value, default: str = "") -> str:
        if pd.isna(value):
            return default
        text = str(value)
        return "" if text.lower() == "nan" else text

    default_actionability = 'suppress' if 'llm_actionability' in alerts.columns else 'actionable'
    alerts['llm_actionability'] = alerts.get(
        'llm_actionability',
        pd.Series(default_actionability, index=alerts.index),
    ).fillna(default_actionability).astype(str)

    score_mask = (
        (pd.to_numeric(alerts['conflict_index'], errors='coerce').fillna(0.0) >= REPORT_MIN_CONFLICT_INDEX)
        & (pd.to_numeric(alerts['innov_z'], errors='coerce').fillna(0.0) >= REPORT_MIN_INNOV_Z)
    )
    actionable_alerts = (
        alerts.loc[score_mask & alerts['llm_actionability'].eq('actionable')]
        .sort_values(['innov_z', 'conflict_index'], ascending=[False, False])
        .head(REPORT_MAX_ALERTS)
        .copy()
    )
    suppressed_alerts = (
        alerts.loc[score_mask & ~alerts['llm_actionability'].eq('actionable')]
        .sort_values(['innov_z', 'conflict_index'], ascending=[False, False])
        .copy()
    )

    result = {
        'date': target_date,
        'generated_at': datetime.now().isoformat(),
        'total_cities': len(today),
        'raw_alert_count': len(alerts),
        'alert_count': len(actionable_alerts),
        'suppressed_count': len(suppressed_alerts),
        'output_policy': {
            'max_alerts': REPORT_MAX_ALERTS,
            'min_conflict_index': REPORT_MIN_CONFLICT_INDEX,
            'min_innovation_z': REPORT_MIN_INNOV_Z,
            'min_exact_support': LLM_MIN_EXACT_SUPPORT,
            'allow_single_article_exact': LLM_ALLOW_SINGLE_ARTICLE_EXACT,
            'allow_strategic_single_support': LLM_ALLOW_STRATEGIC_SINGLE_SUPPORT,
            'allow_roi_prior_single_support': LLM_ALLOW_ROI_PRIOR_SINGLE_SUPPORT,
        },
        'alerts': [
            {
                'city': r['city'],
                'risk_level': int(r['risk_level']),
                'risk_label': r['risk_label'],
                'conflict_index': round(float(r['conflict_index']), 1),
                'innovation_z': round(float(r['innov_z']), 2),
                'guide': r['risk_guide'],
                'events': int(r['events']),
                'lat': None if pd.isna(r.get('lat')) else round(float(r.get('lat')), 4),
                'lon': None if pd.isna(r.get('lon')) else round(float(r.get('lon')), 4),
                'country_code': r.get('country_code', ''),
                'source_url': safe_json_text(r.get('report_source_url', '')),
                'llm_confidence': round(float(r.get('llm_confidence', -1)), 2),
                'llm_reason': safe_json_text(r.get('llm_reason', '')),
                'llm_validation_type': safe_json_text(r.get('llm_validation_type', '')),
                'llm_resolved_city': safe_json_text(r.get('llm_resolved_city', '')),
                'llm_event_summary': safe_json_text(r.get('llm_event_summary', '')),
                'llm_imagery_need': safe_json_text(r.get('llm_imagery_need', '')),
                'llm_actionability': safe_json_text(r.get('llm_actionability', '')),
                'llm_article_count': int(r.get('llm_article_count', 0) or 0),
                'llm_exact_support': int(r.get('llm_exact_support', 0) or 0),
                'llm_nearby_support': int(r.get('llm_nearby_support', 0) or 0),
                'llm_invalid_support': int(r.get('llm_invalid_support', 0) or 0),
                'llm_unclear_count': int(r.get('llm_unclear_count', 0) or 0),
                'llm_strategic_support': int(r.get('llm_strategic_support', 0) or 0),
                'llm_roi_prior_support': int(r.get('llm_roi_prior_support', 0) or 0),
                'llm_evidence_span': safe_json_text(r.get('llm_evidence_span', '')),
                'llm_target_category': safe_json_text(r.get('llm_target_category', '')),
                'llm_keep': bool(r.get('llm_keep', True)),
            } for _, r in actionable_alerts.iterrows()
        ],
        'summary': [
            {
                'city': r['city'],
                'risk_level': int(r['risk_level']),
                'conflict_index': round(float(r['conflict_index']), 1),
                'z_score': round(float(r['innov_z']), 2),
                'source_url': safe_json_text(r.get('report_source_url', '')),
                'llm_validation_type': safe_json_text(r.get('llm_validation_type', '')),
                'llm_resolved_city': safe_json_text(r.get('llm_resolved_city', '')),
                'llm_event_summary': safe_json_text(r.get('llm_event_summary', '')),
                'llm_imagery_need': safe_json_text(r.get('llm_imagery_need', '')),
            } for _, r in today.nlargest(10, 'conflict_index').iterrows()
        ]
    }

    save_path = OUTPUT_DIR / f"{target_date}.json"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  [결과 저장] {save_path.name}")
    return save_path


def run_single_day(target_date: str, fetch: bool = False, use_llm: bool = False):
    """특정 날짜의 파이프라인 전체 실행"""
    print(f"\n--- 파이프라인 가동: {target_date} ---")

    if fetch: fetch_daily(target_date)

    # 데이터 로드 (기존 대용량 Parquet + 수집된 일별 Parquet)
    raw = load_all_data()
    if raw.empty:
        print("  [오류] 분석 가능한 데이터가 없습니다.")
        return

    # 1. 갈등 지수 산출
    city_daily = compute_conflict_index(raw)
    
    # 2. 칼만 필터 기반 이상 징후 탐지
    results = detect_anomalies(city_daily)
    url_df = None

    # 3. LLM 게이트키퍼 검증
    if use_llm:
        try:
            from pipeline.llm_verifier import verify_top_cities, summarize_baseline_cities
        except ModuleNotFoundError:
            print("  [LLM] 검증 모듈이 없어 LLM 단계는 건너뜁니다.")
            results = filter_blacklist(results)
        else:
            url_path = Path(__file__).resolve().parent.parent / 'gdelt_url_final.parquet'
            if url_path.exists():
                url_df = pd.read_parquet(url_path, columns=['GLOBALEVENTID', 'SOURCEURL'])
                
                # --- [수정] Track 1 검증과 Track 2 요약을 각각 실행 ---
                # Track 1: 이상 징후 (is_anomaly == True)
                results = verify_top_cities(results, raw, url_df, target_date)
                
                # Track 2: 전략 거점 요약 (Z-score 탈락자 중 지수 5000 이상만 추려서 요약)
                track2_mask = (results['is_anomaly'] == False) & (results['conflict_index'] >= 5000)
                if track2_mask.any():
                    track2_df = results[track2_mask].copy()
                    # 요약 생성 후 원본에 합치기
                    track2_summarized = summarize_baseline_cities(track2_df, raw, url_df, target_date)
                    results.loc[track2_mask, 'llm_baseline_summary'] = track2_summarized['llm_baseline_summary']
                
            else:
                print("  [LLM] URL 매핑 파일 없음. 블랙리스트만 적용.")
                results = filter_blacklist(results)

    results = attach_report_urls(results, raw, url_df)
    report = format_report(results, target_date)
    print(report)
    save_result(results, target_date)



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
    parser.add_argument('--use-llm', action='store_true', help='LLM 검증 활성화 (기본값: 비활성화)')
    parser.add_argument('--no-llm', action='store_true', help='호환용 옵션. 기본적으로 LLM은 비활성화되어 있습니다.')
    args = parser.parse_args()

    if args.backtest:
        run_backtest()
    else:
        target = args.date or datetime.now().strftime('%Y%m%d')
        run_single_day(target, fetch=args.fetch, use_llm=args.use_llm and not args.no_llm)


if __name__ == '__main__':
    main()
