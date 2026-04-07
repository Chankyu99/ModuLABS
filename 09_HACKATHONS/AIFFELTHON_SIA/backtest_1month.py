import pandas as pd
from pipeline.conflict_index import compute_conflict_index, detect_anomalies
from pipeline.gdelt_fetcher import load_all_data
from pipeline.llm_verifier import verify_top_cities, filter_blacklist
from pipeline.config import OUTPUT_DIR

def run_1month_backtest():
    print("--- 1개월(2/28~3/31) 백테스트 시작 ---")
    raw = load_all_data(include_daily=True)
    city_daily = compute_conflict_index(raw)
    anom = detect_anomalies(city_daily)

    phases = {
        'Phase 1 (2/28~3/10)': ('20260228', '20260310'),
        'Phase 2 (3/11~3/20)': ('20260311', '20260320'),
        'Phase 3 (3/21~3/31)': ('20260321', '20260331'),
    }

    url_df = pd.read_parquet('gdelt_url_final.parquet', columns=['GLOBALEVENTID', 'SOURCEURL'])
    
    with open('1month_backtest_report_temp.md', 'w') as f:
        f.write("# 1개월 전쟁 양상 백테스트 결과 (2/28 ~ 3/31)\n\n")

        for p_name, (start_d, end_d) in phases.items():
            print(f"\n[{p_name}] 분석 중...")
            f.write(f"## {p_name}\n")
            
            p_data = anom[(anom['date'] >= start_d) & (anom['date'] <= end_d) & (anom['is_anomaly'] == True)].copy()
            if p_data.empty:
                f.write("탐지된 핵심 이상 징후 없음.\n\n")
                continue
            
            p_data = filter_blacklist(p_data)
            
            # 각 도시별로 phase 내 최고 Z-score를 기록한 날짜 추출
            idx = p_data.groupby('city')['innov_z'].idxmax()
            top_cities = p_data.loc[idx].sort_values('innov_z', ascending=False).head(7)
            
            f.write("| 순위 | 도시(위험일) | 최대 Z-Score | 총 갈등 지수(I) | LLM 신뢰도 | LLM 사유 |\n")
            f.write("| :---: | :--- | ---: | ---: | :---: | :--- |\n")
            
            for rank, r in enumerate(top_cities.iterrows(), 1):
                idx, row = r
                city = row['city']
                peak_date = row['date']
                
                # LLM Verification for this specific city on its peak date
                # We mock a small anomalies df just for this city+date to pass to verify_top_cities
                sub_anom = anom[(anom['city'] == city) & (anom['date'] == peak_date)].copy()
                sub_anom['is_anomaly'] = True
                
                verified = verify_top_cities(sub_anom, raw, url_df, peak_date)
                
                llm_conf = verified['llm_confidence'].iloc[0] if 'llm_confidence' in verified.columns else -1.0
                llm_reason = verified['llm_reason'].iloc[0] if 'llm_reason' in verified.columns else ""
                
                conf_str = f"**{llm_conf:.0%}**" if llm_conf >= 0 else "N/A"
                if llm_conf < 0.3 and llm_conf >= 0:
                    conf_str = f"⚠️ {conf_str}"
                elif llm_conf >= 0.7:
                    conf_str = f"✅ {conf_str}"
                    
                city_name = city.split(',')[0]
                
                f.write(f"| {rank} | **{city_name}** ({peak_date[4:6]}/{peak_date[6:]}) | {row['innov_z']:.1f} | {row['conflict_index']:.0f} | {conf_str} | {llm_reason} |\n")
                
            f.write("\n---\n")
            
    print("완료되었습니다. '1month_backtest_report_temp.md'를 확인하세요.")

if __name__ == '__main__':
    run_1month_backtest()
