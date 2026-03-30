from pathlib import Path
import duckdb
import pandas as pd

BASE = Path("/Users/son/SSC/projects/SIA")
CORE = BASE / "outputs" / "scenario_C" / "scenario_c_tasking_candidate_core.parquet"
URL = BASE / "gdelt_url_final.parquet"
OUT = BASE / "outputs" / "scenario_C" / "step07_refilter"
OUT.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(BASE / "step07_refilter.duckdb"))

# 1) 뷰 연결
con.execute(f"""
CREATE OR REPLACE VIEW c_core AS
SELECT *
FROM read_parquet('{CORE.as_posix()}');
""")

con.execute(f"""
CREATE OR REPLACE VIEW gdelt_url AS
SELECT *
FROM read_parquet('{URL.as_posix()}');
""")

# 2) 이벤트별 대표 URL 1개
con.execute("""
CREATE OR REPLACE VIEW gdelt_url_lookup AS
SELECT
    GLOBALEVENTID,
    MIN(SOURCEURL) AS SOURCEURL
FROM gdelt_url
GROUP BY 1
""")

# 3) URL 붙이기
con.execute("""
CREATE OR REPLACE TABLE c_core_with_url AS
SELECT
    c.*,
    u.SOURCEURL
FROM c_core c
LEFT JOIN gdelt_url_lookup u USING (GLOBALEVENTID)
""")

# 4) direct triad / IRGC / FARS / city-level flag 생성
con.execute("""
CREATE OR REPLACE TABLE c_core_flags AS
SELECT
    *,

    CASE
        WHEN Actor1CountryCode IN ('IRN','USA','ISR')
         AND Actor2CountryCode IN ('IRN','USA','ISR')
         AND Actor1CountryCode <> Actor2CountryCode
        THEN 1 ELSE 0
    END AS direct_triad_pair_flag,

    CASE
        WHEN lower(coalesce(Actor1Name,'')) LIKE '%irgc%'
          OR lower(coalesce(Actor1Name,'')) LIKE '%revolutionary guard%'
          OR lower(coalesce(Actor1Name,'')) LIKE '%islamic revolutionary guard%'
          OR lower(coalesce(Actor1Name,'')) LIKE '%pasdaran%'
          OR lower(coalesce(Actor2Name,'')) LIKE '%irgc%'
          OR lower(coalesce(Actor2Name,'')) LIKE '%revolutionary guard%'
          OR lower(coalesce(Actor2Name,'')) LIKE '%islamic revolutionary guard%'
          OR lower(coalesce(Actor2Name,'')) LIKE '%pasdaran%'
        THEN 1 ELSE 0
    END AS irgc_actor_flag,

    CASE
        WHEN lower(coalesce(SOURCEURL,'')) LIKE '%farsnews%'
          OR lower(coalesce(SOURCEURL,'')) LIKE '%farsnews.ir%'
        THEN 1 ELSE 0
    END AS fars_source_flag,

    CASE
        WHEN ActionGeo_Type = 4
         AND ActionGeo_FeatureID IS NOT NULL
         AND ActionGeo_FeatureID <> ''
         AND lower(trim(coalesce(ActionGeo_FullName,''))) NOT IN (
             'iran','israel','iraq','syria','lebanon',
             'saudi arabia','united states','palestine'
         )
        THEN 1 ELSE 0
    END AS city_level_flag

FROM c_core_with_url
""")

# 5) direct triad city-level만 남기기
con.execute("""
CREATE OR REPLACE TABLE c_core_direct_v1 AS
SELECT *
FROM c_core_flags
WHERE city_level_flag = 1
  AND direct_triad_pair_flag = 1
""")

# 6) IRGC / FARS 검토용 별도 테이블
con.execute("""
CREATE OR REPLACE TABLE c_core_irgc_fars_review_v1 AS
SELECT *
FROM c_core_flags
WHERE city_level_flag = 1
  AND (irgc_actor_flag = 1 OR fars_source_flag = 1)
""")

# 7) counts
counts = con.sql("""
SELECT 'c_core_input' AS stage, COUNT(*) AS cnt FROM c_core
UNION ALL
SELECT 'with_url', COUNT(*) FROM c_core_with_url
UNION ALL
SELECT 'flags_built', COUNT(*) FROM c_core_flags
UNION ALL
SELECT 'city_level', COUNT(*) FROM c_core_flags WHERE city_level_flag = 1
UNION ALL
SELECT 'direct_v1', COUNT(*) FROM c_core_direct_v1
UNION ALL
SELECT 'irgc_fars_review', COUNT(*) FROM c_core_irgc_fars_review_v1
""").df()
counts.to_csv(OUT / "step07_counts.csv", index=False)

# 8) flag stats
flag_stats = con.sql("""
SELECT
    SUM(direct_triad_pair_flag) AS direct_triad_pair_rows,
    SUM(irgc_actor_flag) AS irgc_actor_rows,
    SUM(fars_source_flag) AS fars_source_rows,
    SUM(city_level_flag) AS city_level_rows
FROM c_core_flags
""").df()
flag_stats.to_csv(OUT / "step07_flag_stats.csv", index=False)

# 9) direct hotspot
hotspots = con.sql("""
SELECT
    ActionGeo_FeatureID,
    ActionGeo_FullName,
    ActionGeo_Lat,
    ActionGeo_Long,
    COUNT(*) AS event_cnt,
    SUM(COALESCE(NumMentions,0)) AS mention_sum,
    AVG(COALESCE(GoldsteinScale,0)) AS avg_goldstein,
    AVG(COALESCE(AvgTone,0)) AS avg_tone
FROM c_core_direct_v1
GROUP BY 1,2,3,4
ORDER BY event_cnt DESC, mention_sum DESC
LIMIT 30
""").df()
hotspots.to_csv(OUT / "step07_hotspots_direct_v1.csv", index=False)

# 10) 대표 샘플
sample = con.sql("""
SELECT
    SQLDATE,
    ActionGeo_FullName,
    Actor1Name, Actor1CountryCode,
    Actor2Name, Actor2CountryCode,
    EventCode, EventRootCode,
    GoldsteinScale,
    NumMentions,
    direct_triad_pair_flag,
    irgc_actor_flag,
    fars_source_flag,
    SOURCEURL
FROM c_core_direct_v1
ORDER BY NumMentions DESC
LIMIT 100
""").df()
sample.to_csv(OUT / "step07_sample_direct_v1.csv", index=False)

# 11) IRGC/FARS 검토 샘플
review = con.sql("""
SELECT
    SQLDATE,
    ActionGeo_FullName,
    Actor1Name, Actor1CountryCode,
    Actor2Name, Actor2CountryCode,
    EventCode, EventRootCode,
    GoldsteinScale,
    NumMentions,
    irgc_actor_flag,
    fars_source_flag,
    SOURCEURL
FROM c_core_irgc_fars_review_v1
ORDER BY NumMentions DESC
LIMIT 100
""").df()
review.to_csv(OUT / "step07_irgc_fars_review.csv", index=False)

# 12) 저장
con.execute(f"COPY c_core_direct_v1 TO '{(OUT / 'c_core_direct_v1.parquet').as_posix()}' (FORMAT parquet)")
con.execute(f"COPY c_core_irgc_fars_review_v1 TO '{(OUT / 'c_core_irgc_fars_review_v1.parquet').as_posix()}' (FORMAT parquet)")

print("완료:", OUT)