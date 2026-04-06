"""
7일 백테스트 (2/28~3/6) + 상위 도시별 SOURCEURL 추출
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from pipeline.gdelt_fetcher import load_all_data
from pipeline.conflict_index import compute_conflict_index, detect_anomalies
from pipeline.config import CONFIRMED_CODES, MONITORED_COUNTRIES

print("데이터 로딩...")
raw = load_all_data()

# SOURCEURL 매핑 테이블 로드
print("URL 매핑 테이블 로딩...")
url_df = pd.read_parquet('gdelt_url_final.parquet', columns=['GLOBALEVENTID', 'SOURCEURL'])

# 갈등 지수 산출 및 이상 탐지
city_daily = compute_conflict_index(raw)
results = detect_anomalies(city_daily)

# 필터링된 원본 데이터 (GLOBALEVENTID 포함)
mask = (
    (raw['Actor1CountryCode'].isin(MONITORED_COUNTRIES) | 
     raw['Actor2CountryCode'].isin(MONITORED_COUNTRIES)) &
    raw['EventCode'].astype(str).str.split('.').str[0].isin(CONFIRMED_CODES) &
    (raw['ActionGeo_Type'] == 4)
)
filtered = raw[mask].copy()
filtered['date'] = filtered['SQLDATE'].astype(str).str[:8]

# 날짜 범위
dates = ['20260228', '20260301', '20260302', '20260303', '20260304', '20260305', '20260306']

print("\n" + "=" * 100)
print("  🛰️  SIA 7일 백테스트 (2026-02-28 ~ 2026-03-06)")
print("=" * 100)

all_rows = []

for date in dates:
    day = results[(results['date'] == date) & (results['is_anomaly'] == True)]
    if day.empty:
        print(f"\n📅 {date}: 이상 징후 없음")
        continue
    
    top3 = day.sort_values('innov_z', ascending=False).head(3)
    
    print(f"\n{'─'*100}")
    print(f"📅 {date} — 이상 징후 {len(day)}개 도시 (상위 3개)")
    print(f"{'─'*100}")
    
    for _, row in top3.iterrows():
        city = row['city']
        city_short = city.split(',')[0][:25]
        
        print(f"\n  {row['risk_emoji']} {row['risk_label']} | {city_short} | I={row['conflict_index']:.0f} | Z={row['innov_z']:.1f}")
        
        # 해당 날짜+도시의 GLOBALEVENTID 추출
        city_events = filtered[
            (filtered['date'] == date) & 
            (filtered['ActionGeo_FullName'] == city)
        ]
        
        if city_events.empty:
            print(f"     → 이벤트 없음 (0 채우기 영향)")
            continue
        
        # NumSources 상위 4개 이벤트의 SOURCEURL
        top_events = city_events.nlargest(4, 'NumSources')
        event_ids = top_events['GLOBALEVENTID'].values
        
        urls = url_df[url_df['GLOBALEVENTID'].isin(event_ids)].drop_duplicates('GLOBALEVENTID')
        merged = top_events[['GLOBALEVENTID', 'NumMentions', 'NumSources', 'AvgTone']].merge(
            urls, on='GLOBALEVENTID', how='left'
        )
        
        for _, u in merged.iterrows():
            url = u.get('SOURCEURL', 'N/A')
            if pd.isna(url): url = 'N/A'
            print(f"     📰 Sources={u['NumSources']:.0f} Mentions={u['NumMentions']:.0f} Tone={u['AvgTone']:.1f}")
            print(f"        {url}")
        
        # 아티팩트용 데이터 수집
        for _, u in merged.iterrows():
            all_rows.append({
                'date': date,
                'city': city_short,
                'risk': row['risk_label'],
                'I': round(row['conflict_index']),
                'Z': round(row['innov_z'], 1),
                'sources': int(u.get('NumSources', 0)),
                'mentions': int(u.get('NumMentions', 0)),
                'tone': round(u.get('AvgTone', 0), 1),
                'url': u.get('SOURCEURL', 'N/A'),
            })

print(f"\n{'='*100}")
print(f"  백테스트 완료")
print(f"{'='*100}")

# CSV 저장
if all_rows:
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv('output/backtest_7day_urls.csv', index=False, encoding='utf-8-sig')
    print(f"\n  [저장] output/backtest_7day_urls.csv ({len(df_out)}건)")
