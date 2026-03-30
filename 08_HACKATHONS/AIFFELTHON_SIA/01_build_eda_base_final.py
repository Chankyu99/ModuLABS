from pathlib import Path
import duckdb
import pandas as pd

BASE = Path("/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA")
MAIN = BASE / "gdelt_main_final.parquet"
URL = BASE / "gdelt_url_final.parquet"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

con = duckdb.connect(str(BASE / "gdelt_eda.duckdb"))

# 1) 원본 뷰 연결
con.execute(f"""
CREATE OR REPLACE VIEW gdelt_main AS
SELECT *
FROM read_parquet('{MAIN.as_posix()}');
""")

con.execute(f"""
CREATE OR REPLACE VIEW gdelt_url AS
SELECT *
FROM read_parquet('{URL.as_posix()}');
""")

# 2) 스키마 저장
schema_main = con.sql("DESCRIBE SELECT * FROM gdelt_main").df()
schema_url = con.sql("DESCRIBE SELECT * FROM gdelt_url").df()

schema_main.to_csv(OUT / "schema_main.csv", index=False)
schema_url.to_csv(OUT / "schema_url.csv", index=False)

# 3) 품질 확인
quality_main = con.sql("""
SELECT
    COUNT(*) AS total_rows,
    MIN(SQLDATE) AS min_sqldate,
    MAX(SQLDATE) AS max_sqldate,
    SUM(CASE WHEN EventCode IS NULL THEN 1 ELSE 0 END) AS eventcode_null,
    SUM(CASE WHEN EventRootCode IS NULL THEN 1 ELSE 0 END) AS eventroot_null,
    SUM(CASE WHEN GoldsteinScale IS NULL THEN 1 ELSE 0 END) AS goldstein_null,
    SUM(CASE WHEN NumMentions IS NULL THEN 1 ELSE 0 END) AS nummentions_null,
    SUM(CASE WHEN ActionGeo_Lat IS NULL OR ActionGeo_Long IS NULL THEN 1 ELSE 0 END) AS geo_null,
    SUM(CASE WHEN ActionGeo_FeatureID IS NULL OR ActionGeo_FeatureID = '' THEN 1 ELSE 0 END) AS featureid_null
FROM gdelt_main
""").df()

quality_url = con.sql("""
SELECT
    COUNT(*) AS url_rows,
    COUNT(DISTINCT GLOBALEVENTID) AS url_distinct_event_ids,
    SUM(CASE WHEN SOURCEURL IS NULL OR SOURCEURL = '' THEN 1 ELSE 0 END) AS sourceurl_null
FROM gdelt_url
""").df()

join_check = con.sql("""
SELECT
    COUNT(*) AS main_rows,
    COUNT(u.GLOBALEVENTID) AS matched_url_rows,
    COUNT(DISTINCT m.GLOBALEVENTID) AS main_distinct_ids,
    COUNT(DISTINCT u.GLOBALEVENTID) AS matched_distinct_ids
FROM gdelt_main m
LEFT JOIN gdelt_url u USING (GLOBALEVENTID)
""").df()

quality_main.to_csv(OUT / "quality_main.csv", index=False)
quality_url.to_csv(OUT / "quality_url.csv", index=False)
join_check.to_csv(OUT / "join_check.csv", index=False)

# 4) broad slice
con.execute("""
CREATE OR REPLACE TABLE triad_broad AS
SELECT
    GLOBALEVENTID,
    SQLDATE,
    Actor1Name, Actor1CountryCode,
    Actor2Name, Actor2CountryCode,
    IsRootEvent,
    EventCode, EventRootCode,
    QuadClass, GoldsteinScale,
    NumMentions, AvgTone,
    ActionGeo_Type, ActionGeo_FullName,
    ActionGeo_CountryCode,
    ActionGeo_Lat, ActionGeo_Long, ActionGeo_FeatureID
FROM gdelt_main
WHERE SQLDATE BETWEEN 20130401 AND 20260326
  AND (
       Actor1CountryCode IN ('IRN','USA','ISR')
    OR Actor2CountryCode IN ('IRN','USA','ISR')
    OR (ActionGeo_Lat BETWEEN 12 AND 42 AND ActionGeo_Long BETWEEN 25 AND 65)
  )
  AND ActionGeo_Lat IS NOT NULL
  AND ActionGeo_Long IS NOT NULL
""")

# 5) combat slice
con.execute("""
CREATE OR REPLACE TABLE triad_combat AS
SELECT *
FROM triad_broad
WHERE QuadClass = 4
  AND EventRootCode IN ('18','19','20')
""")

# 6) tasking candidate
con.execute("""
CREATE OR REPLACE TABLE triad_tasking_candidate AS
SELECT *
FROM triad_combat
WHERE IsRootEvent = 1
  AND GoldsteinScale < -7
  AND ActionGeo_Type = 4
""")

# 7) retention
retention = con.sql("""
SELECT 'broad' AS stage, COUNT(*) AS cnt FROM triad_broad
UNION ALL
SELECT 'combat', COUNT(*) FROM triad_combat
UNION ALL
SELECT 'tasking_candidate', COUNT(*) FROM triad_tasking_candidate
""").df()
retention.to_csv(OUT / "filter_retention.csv", index=False)

# 8) event / dyad / hotspot
event_root_distribution = con.sql("""
SELECT EventRootCode, COUNT(*) AS cnt
FROM triad_broad
GROUP BY 1
ORDER BY cnt DESC
""").df()
event_root_distribution.to_csv(OUT / "event_root_distribution.csv", index=False)

event_code_top20 = con.sql("""
SELECT EventCode, COUNT(*) AS cnt
FROM triad_broad
GROUP BY 1
ORDER BY cnt DESC
LIMIT 20
""").df()
event_code_top20.to_csv(OUT / "event_code_top20.csv", index=False)

dyad_distribution = con.sql("""
SELECT
    Actor1CountryCode,
    Actor2CountryCode,
    COUNT(*) AS cnt
FROM triad_broad
GROUP BY 1,2
ORDER BY cnt DESC
""").df()
dyad_distribution.to_csv(OUT / "dyad_distribution.csv", index=False)

# combat hotspot
hotspots_combat_top20 = con.sql("""
SELECT
    ActionGeo_FeatureID,
    ActionGeo_FullName,
    ActionGeo_Lat,
    ActionGeo_Long,
    COUNT(*) AS event_cnt,
    SUM(COALESCE(NumMentions,0)) AS mention_sum,
    AVG(COALESCE(GoldsteinScale,0)) AS avg_goldstein,
    AVG(COALESCE(AvgTone,0)) AS avg_tone
FROM triad_combat
GROUP BY 1,2,3,4
ORDER BY event_cnt DESC, mention_sum DESC
LIMIT 20
""").df()
hotspots_combat_top20.to_csv(OUT / "hotspots_combat_top20.csv", index=False)

# candidate hotspot (최종 ROI에 더 가까움)
hotspots_candidate_top20 = con.sql("""
SELECT
    ActionGeo_FeatureID,
    ActionGeo_FullName,
    ActionGeo_Lat,
    ActionGeo_Long,
    COUNT(*) AS event_cnt,
    SUM(COALESCE(NumMentions,0)) AS mention_sum,
    AVG(COALESCE(GoldsteinScale,0)) AS avg_goldstein,
    AVG(COALESCE(AvgTone,0)) AS avg_tone
FROM triad_tasking_candidate
GROUP BY 1,2,3,4
ORDER BY event_cnt DESC, mention_sum DESC
LIMIT 20
""").df()
hotspots_candidate_top20.to_csv(OUT / "hotspots_candidate_top20.csv", index=False)

# yearly
yearly_counts = con.sql("""
SELECT
    CAST(SUBSTR(CAST(SQLDATE AS VARCHAR), 1, 4) AS INTEGER) AS year,
    COUNT(*) AS event_cnt,
    SUM(COALESCE(NumMentions,0)) AS mention_sum,
    AVG(COALESCE(GoldsteinScale,0)) AS avg_goldstein
FROM triad_combat
GROUP BY 1
ORDER BY 1
""").df()
yearly_counts.to_csv(OUT / "yearly_counts.csv", index=False)

# 9) URL join
con.execute("""
CREATE OR REPLACE TABLE triad_candidate_with_url AS
SELECT
    c.*,
    u.SOURCEURL
FROM triad_tasking_candidate c
LEFT JOIN gdelt_url u USING (GLOBALEVENTID)
""")

# raw ground truth pool
ground_truth_candidates = con.sql("""
SELECT
    SQLDATE,
    GLOBALEVENTID,
    ActionGeo_FullName,
    ActionGeo_FeatureID,
    ActionGeo_Lat,
    ActionGeo_Long,
    Actor1Name, Actor1CountryCode,
    Actor2Name, Actor2CountryCode,
    EventCode, EventRootCode,
    GoldsteinScale,
    NumMentions,
    AvgTone,
    SOURCEURL
FROM triad_candidate_with_url
WHERE SOURCEURL IS NOT NULL
  AND SOURCEURL <> ''
ORDER BY NumMentions DESC
LIMIT 500
""").df()
ground_truth_candidates.to_csv(OUT / "ground_truth_candidates.csv", index=False)

# dedupe ground truth
ground_truth_candidates_dedup = con.sql("""
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY SQLDATE, ActionGeo_FeatureID, EventCode
               ORDER BY NumMentions DESC
           ) AS rn
    FROM triad_candidate_with_url
    WHERE SOURCEURL IS NOT NULL
      AND SOURCEURL <> ''
)
SELECT
    SQLDATE,
    GLOBALEVENTID,
    ActionGeo_FullName,
    ActionGeo_FeatureID,
    ActionGeo_Lat,
    ActionGeo_Long,
    Actor1Name, Actor1CountryCode,
    Actor2Name, Actor2CountryCode,
    EventCode, EventRootCode,
    GoldsteinScale,
    NumMentions,
    AvgTone,
    SOURCEURL
FROM ranked
WHERE rn = 1
ORDER BY NumMentions DESC
LIMIT 200
""").df()
ground_truth_candidates_dedup.to_csv(OUT / "ground_truth_candidates_dedup.csv", index=False)

# 10) 저장
con.execute(f"COPY triad_broad TO '{(OUT / 'triad_broad.parquet').as_posix()}' (FORMAT parquet)")
con.execute(f"COPY triad_combat TO '{(OUT / 'triad_combat.parquet').as_posix()}' (FORMAT parquet)")
con.execute(f"COPY triad_tasking_candidate TO '{(OUT / 'triad_tasking_candidate.parquet').as_posix()}' (FORMAT parquet)")

print("완료: outputs 폴더 확인")
print("주의: NumSources / NumArticles / EventBaseCode는 현재 스키마에 없음")