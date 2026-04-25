"""
GT 각 (date, city)을 A/C 이분법으로 분류.

티어 정의:
  C : mentions_on_date < 10  OR  active_days < 10
      → GDELT 관측 부재 or Kalman 초기화 불가 → 평가 제외
  A : mentions_on_date >= 10 AND active_days >= 10
      → 평가 대상

mentions_on_date: GT날 해당 도시 GDELT rows의 NumMentions 합계.
  이름 lookup 우선순위:
    1) GT 이름 exact match
    2) FeatureID alias (같은 FeatureID 아래 다른 표기, e.g. Esfahan↔Isfahan)
    3) substring match
    4) geo fallback (50km 이내 최다 mentions 도시)

active_days: GT날 이전 30일 중 해당 도시 이름으로 이벤트가 있는 날 수.
"""

import os
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

GT_DIR    = PROJECT_ROOT / "data" / "ground_truth"
OUT_PATH  = PROJECT_ROOT / "gt_detectability.csv"

MIN_MENTIONS   = 10
MIN_ACTIVE_DAYS = 10


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p)*np.cos(lat2*p)*(1-np.cos((lon2-lon1)*p))/2
    return 12742 * np.arcsin(np.sqrt(a))


def load_gt():
    rows = []
    for f in sorted(os.listdir(GT_DIR)):
        if not f.endswith(".csv"): continue
        df = pd.read_csv(GT_DIR / f)
        for _, r in df.iterrows():
            rows.append({'date': f[:8], 'city': str(r['ActionGeo_FullName']).strip(),
                         'lat': r.get('Lat'), 'lon': r.get('Long')})
    return pd.DataFrame(rows)


def get_mentions(gt_row, same_day: pd.DataFrame, name_to_fid: dict) -> tuple:
    """Returns (mentions, match_type)."""
    cl = gt_row['city'].lower().strip()

    # 1) exact
    hits = same_day[same_day['cl'] == cl]
    if not hits.empty:
        return int(hits['NumMentions'].sum()), 'exact'

    # 2) FeatureID alias
    fid = name_to_fid.get(cl)
    if fid is not None:
        hits = same_day[same_day['ActionGeo_FeatureID'] == fid]
        if not hits.empty:
            return int(hits['NumMentions'].sum()), 'alias'

    # 3) substring
    hits = same_day[same_day['cl'].str.contains(cl, na=False, regex=False)]
    if not hits.empty:
        return int(hits['NumMentions'].sum()), 'substring'

    # 4) geo fallback (50km)
    if pd.notna(gt_row['lat']) and pd.notna(gt_row['lon']):
        lat0, lon0 = float(gt_row['lat']), float(gt_row['lon'])
        bb = same_day[same_day['ActionGeo_Lat'].between(lat0-0.6, lat0+0.6) &
                      same_day['ActionGeo_Long'].between(lon0-0.6, lon0+0.6)]
        if not bb.empty:
            d = hav(lat0, lon0, bb['ActionGeo_Lat'].values, bb['ActionGeo_Long'].values)
            near = bb[d <= 50]
            if not near.empty:
                return int(near['NumMentions'].sum()), 'geo'

    return 0, 'none'


def get_active_days(gt_row, raw: pd.DataFrame, name_to_fid: dict) -> int:
    cl = gt_row['city'].lower().strip()
    date = gt_row['date']
    from datetime import datetime, timedelta
    dt = datetime.strptime(date, '%Y%m%d')
    window_start = (dt - timedelta(days=30)).strftime('%Y%m%d')

    prior = raw[(raw['date'] >= window_start) & (raw['date'] < date)]

    # exact or alias or substring
    hits = prior[prior['cl'] == cl]
    if hits.empty:
        fid = name_to_fid.get(cl)
        if fid is not None:
            hits = prior[prior['ActionGeo_FeatureID'] == fid]
    if hits.empty:
        hits = prior[prior['cl'].str.contains(cl, na=False, regex=False)]

    return int(hits.groupby('date').ngroups) if not hits.empty else 0


def main():
    print("[1/3] Loading GDELT...")
    raw = pd.read_parquet(PROJECT_ROOT / "data" / "gdelt_main_2026.parquet")
    raw['cl']   = raw['ActionGeo_FullName'].astype(str).str.lower().str.strip()
    raw['date'] = raw['SQLDATE'].astype(str)

    # name → FeatureID map (using most frequent FID per name)
    name_to_fid = (raw.groupby('cl')['ActionGeo_FeatureID']
                   .agg(lambda x: x.value_counts().index[0])
                   .to_dict())

    print("[2/3] Computing GT tiers...")
    gt = load_gt()
    records = []
    for _, r in gt.iterrows():
        same_day = raw[raw['date'] == r['date']]
        mentions, mt = get_mentions(r, same_day, name_to_fid)
        active = get_active_days(r, raw, name_to_fid)
        tier = 'A' if mentions >= MIN_MENTIONS else 'C'
        records.append({
            'date': r['date'], 'city': r['city'],
            'lat': r['lat'], 'lon': r['lon'],
            'mentions_on_date': mentions,
            'active_days': active,
            'match_type': mt,
            'tier': tier,
        })

    out = pd.DataFrame(records)
    out.to_csv(OUT_PATH, index=False)
    print(f"\n=== Tier distribution (GT n={len(out)}) ===")
    print(out['tier'].value_counts().sort_index())
    print(f"\n=== Match type ===")
    print(out['match_type'].value_counts())
    print(f"\n=== C tier cities ===")
    print(out[out['tier']=='C'][['date','city','mentions_on_date','active_days','match_type']].to_string(index=False))
    print(f"\n→ saved {OUT_PATH}")


if __name__ == "__main__":
    main()
