from pathlib import Path
import sys
import duckdb
import pandas as pd

# -----------------------------
# 설정
# -----------------------------
BASE = Path("/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA")
MAIN = BASE / "gdelt_main_final.parquet"
URL = BASE / "gdelt_url_final.parquet"

if len(sys.argv) != 2 or sys.argv[1].upper() not in {"B", "C"}:
    raise SystemExit("사용법: python 04_run_scenario_compare.py [B|C]")

SCENARIO = sys.argv[1].upper()
OUT = BASE / "outputs" / f"scenario_{SCENARIO}"
OUT.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(BASE / f"gdelt_scenario_{SCENARIO}.duckdb"))

# -----------------------------
# 뷰 연결
# -----------------------------
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

# URL lookup: 이벤트별 대표 URL 1개만 유지
con.execute("""
CREATE OR REPLACE VIEW gdelt_url_lookup AS
SELECT
    GLOBALEVENTID,
    MIN(SOURCEURL) AS SOURCEURL
FROM gdelt_url
GROUP BY 1
""")

# -----------------------------
# 스키마 체크
# -----------------------------
required_cols = {
    "GLOBALEVENTID", "SQLDATE",
    "Actor1Name", "Actor1CountryCode",
    "Actor2Name", "Actor2CountryCode",
    "IsRootEvent",
    "EventCode", "EventRootCode",
    "QuadClass", "GoldsteinScale",
    "NumMentions", "AvgTone",
    "ActionGeo_Type", "ActionGeo_FullName",
    "ActionGeo_CountryCode", "ActionGeo_Lat",
    "ActionGeo_Long", "ActionGeo_FeatureID",
}
schema_main = con.sql("DESCRIBE SELECT * FROM gdelt_main").df()
main_cols = set(schema_main["column_name"].tolist())
missing = sorted(required_cols - main_cols)
if missing:
    raise ValueError(f"main 파일 필수 컬럼 누락: {missing}")

schema_main.to_csv(OUT / "schema_main.csv", index=False)

# -----------------------------
# 공통 SQL 조각
# -----------------------------
BASE_SELECT = """
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
"""

BASE_BROAD_WHERE = """
SQLDATE BETWEEN 20130401 AND 20260326
AND (
     Actor1CountryCode IN ('IRN','USA','ISR')
  OR Actor2CountryCode IN ('IRN','USA','ISR')
  OR (ActionGeo_Lat BETWEEN 12 AND 42 AND ActionGeo_Long BETWEEN 25 AND 65)
)
AND ActionGeo_Lat IS NOT NULL
AND ActionGeo_Long IS NOT NULL
"""

# B안: ISR 전면 제외
FULL_ISR_GEO_EXCLUDE = """
(
    COALESCE(ActionGeo_CountryCode, '') IN ('IS','GZ','WE')
    OR lower(COALESCE(ActionGeo_FullName, '')) LIKE '%israel%'
    OR lower(COALESCE(ActionGeo_FullName, '')) LIKE '%gaza%'
    OR lower(COALESCE(ActionGeo_FullName, '')) LIKE '%jerusalem%'
    OR lower(COALESCE(ActionGeo_FullName, '')) LIKE '%tel aviv%'
    OR lower(COALESCE(ActionGeo_FullName, '')) LIKE '%hebron%'
    OR lower(COALESCE(ActionGeo_FullName, '')) LIKE '%ramallah%'
    OR lower(COALESCE(ActionGeo_FullName, '')) LIKE '%jenin%'
    OR lower(COALESCE(ActionGeo_FullName, '')) LIKE '%bethlehem%'
    OR lower(COALESCE(ActionGeo_FullName, '')) LIKE '%nablus%'
    OR lower(COALESCE(ActionGeo_FullName, '')) LIKE '%west bank%'
)
"""

# C안: 핀셋 보정
PINSET_LOCAL_GEO = """
(
    COALESCE(ActionGeo_CountryCode, '') IN ('GZ','WE')
    OR lower(COALESCE(ActionGeo_FullName, '')) LIKE '%gaza%'
    OR lower(COALESCE(ActionGeo_FullName, '')) LIKE '%jerusalem%'
    OR lower(COALESCE(ActionGeo_FullName, '')) LIKE '%hebron%'
    OR lower(COALESCE(ActionGeo_FullName, '')) LIKE '%ramallah%'
    OR lower(COALESCE(ActionGeo_FullName, '')) LIKE '%jenin%'
    OR lower(COALESCE(ActionGeo_FullName, '')) LIKE '%bethlehem%'
    OR lower(COALESCE(ActionGeo_FullName, '')) LIKE '%nablus%'
    OR lower(COALESCE(ActionGeo_FullName, '')) LIKE '%west bank%'
)
"""

TRIAD_DIRECT = """
(
    COALESCE(Actor1CountryCode, '') IN ('IRN','USA')
    OR COALESCE(Actor2CountryCode, '') IN ('IRN','USA')
)
"""

def fix_swapped_gt_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    혹시 ActionGeo_FeatureID와 EventCode가 뒤바뀐 출력이 생기면 자동 보정.
    """
    if "ActionGeo_FeatureID" not in df.columns or "EventCode" not in df.columns:
        return df

    feat = pd.to_numeric(df["ActionGeo_FeatureID"], errors="coerce")
    evt = pd.to_numeric(df["EventCode"], errors="coerce")

    if feat.notna().sum() == 0 or evt.notna().sum() == 0:
        return df

    feat_looks_like_eventcode = feat.dropna().between(100, 9999).mean()
    evt_looks_like_featureid = (evt.dropna().abs() > 10000).mean()

    if feat_looks_like_eventcode > 0.7 and evt_looks_like_featureid > 0.7:
        df = df.copy()
        df[["ActionGeo_FeatureID", "EventCode"]] = df[["EventCode", "ActionGeo_FeatureID"]]
    return df

# -----------------------------
# 시나리오별 실행
# -----------------------------
if SCENARIO == "B":
    notes = [
        "B안 = ISR 전면 제외",
        "Actor1/2CountryCode='ISR' 행 제거",
        "Israel / Gaza / West Bank / Jerusalem 계열 geo도 제거",
        "목적: 편향이 얼마나 줄어드는지 hard exclude 비교"
    ]

    con.execute(f"""
    CREATE OR REPLACE TABLE scenario_b_broad AS
    SELECT
    {BASE_SELECT}
    FROM gdelt_main
    WHERE {BASE_BROAD_WHERE}
      AND COALESCE(Actor1CountryCode, '') <> 'ISR'
      AND COALESCE(Actor2CountryCode, '') <> 'ISR'
      AND NOT {FULL_ISR_GEO_EXCLUDE}
    """)

    con.execute("""
    CREATE OR REPLACE TABLE scenario_b_combat AS
    SELECT *
    FROM scenario_b_broad
    WHERE QuadClass = 4
      AND EventRootCode IN ('18','19','20')
    """)

    con.execute("""
    CREATE OR REPLACE TABLE scenario_b_tasking_candidate AS
    SELECT *
    FROM scenario_b_combat
    WHERE IsRootEvent = 1
      AND GoldsteinScale < -7
      AND ActionGeo_Type = 4
    """)

    con.execute("""
    CREATE OR REPLACE TABLE scenario_b_candidate_with_url AS
    SELECT
        c.*,
        u.SOURCEURL
    FROM scenario_b_tasking_candidate c
    LEFT JOIN gdelt_url_lookup u USING (GLOBALEVENTID)
    """)

    retention = con.sql("""
    SELECT 'broad' AS stage, COUNT(*) AS cnt FROM scenario_b_broad
    UNION ALL
    SELECT 'combat', COUNT(*) FROM scenario_b_combat
    UNION ALL
    SELECT 'tasking_candidate', COUNT(*) FROM scenario_b_tasking_candidate
    """).df()
    retention.to_csv(OUT / "filter_retention.csv", index=False)

    event_root_distribution = con.sql("""
    SELECT EventRootCode, COUNT(*) AS cnt
    FROM scenario_b_broad
    GROUP BY 1
    ORDER BY cnt DESC
    """).df()
    event_root_distribution.to_csv(OUT / "event_root_distribution.csv", index=False)

    event_code_top20 = con.sql("""
    SELECT EventCode, COUNT(*) AS cnt
    FROM scenario_b_broad
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
    FROM scenario_b_broad
    GROUP BY 1,2
    ORDER BY cnt DESC
    """).df()
    dyad_distribution.to_csv(OUT / "dyad_distribution.csv", index=False)

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
    FROM scenario_b_tasking_candidate
    GROUP BY 1,2,3,4
    ORDER BY event_cnt DESC, mention_sum DESC
    LIMIT 20
    """).df()
    hotspots.to_csv(OUT / "hotspots_candidate_top20.csv", index=False)

    yearly_counts = con.sql("""
    SELECT
        CAST(SUBSTR(CAST(SQLDATE AS VARCHAR), 1, 4) AS INTEGER) AS year,
        COUNT(*) AS event_cnt,
        SUM(COALESCE(NumMentions,0)) AS mention_sum,
        AVG(COALESCE(GoldsteinScale,0)) AS avg_goldstein
    FROM scenario_b_combat
    GROUP BY 1
    ORDER BY 1
    """).df()
    yearly_counts.to_csv(OUT / "yearly_counts.csv", index=False)

    gt = con.sql("""
    WITH ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY SQLDATE, ActionGeo_FeatureID, EventCode
                   ORDER BY NumMentions DESC
               ) AS rn
        FROM scenario_b_candidate_with_url
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

    gt = fix_swapped_gt_columns(gt)
    gt.to_csv(OUT / "ground_truth_candidates_dedup.csv", index=False)

    con.execute(f"COPY scenario_b_broad TO '{(OUT / 'scenario_b_broad.parquet').as_posix()}' (FORMAT parquet)")
    con.execute(f"COPY scenario_b_combat TO '{(OUT / 'scenario_b_combat.parquet').as_posix()}' (FORMAT parquet)")
    con.execute(f"COPY scenario_b_tasking_candidate TO '{(OUT / 'scenario_b_tasking_candidate.parquet').as_posix()}' (FORMAT parquet)")

else:
    notes = [
        "C안 = 핀셋 보정",
        "ISR 전체는 유지",
        "단, Gaza/Jerusalem/West Bank 로컬 사건 중 IRN/USA 직접 actor 연결이 약한 건만 holdout으로 분리",
        "목적: triad 구조는 보존하면서 Gaza/Jerusalem 과증폭만 줄이기"
    ]

    con.execute(f"""
    CREATE OR REPLACE TABLE scenario_c_broad AS
    SELECT
    {BASE_SELECT}
    FROM gdelt_main
    WHERE {BASE_BROAD_WHERE}
    """)

    con.execute("""
    CREATE OR REPLACE TABLE scenario_c_combat AS
    SELECT *
    FROM scenario_c_broad
    WHERE QuadClass = 4
      AND EventRootCode IN ('18','19','20')
    """)

    con.execute("""
    CREATE OR REPLACE TABLE scenario_c_tasking_candidate_all AS
    SELECT *
    FROM scenario_c_combat
    WHERE IsRootEvent = 1
      AND GoldsteinScale < -7
      AND ActionGeo_Type = 4
    """)

    con.execute(f"""
    CREATE OR REPLACE TABLE scenario_c_tasking_candidate_holdout AS
    SELECT *,
           1 AS pinset_exclude_flag
    FROM scenario_c_tasking_candidate_all
    WHERE {PINSET_LOCAL_GEO}
      AND NOT {TRIAD_DIRECT}
    """)

    con.execute(f"""
    CREATE OR REPLACE TABLE scenario_c_tasking_candidate_core AS
    SELECT *,
           CASE
               WHEN {PINSET_LOCAL_GEO} AND NOT {TRIAD_DIRECT} THEN 1
               ELSE 0
           END AS pinset_exclude_flag
    FROM scenario_c_tasking_candidate_all
    WHERE NOT ({PINSET_LOCAL_GEO} AND NOT {TRIAD_DIRECT})
    """)

    con.execute("""
    CREATE OR REPLACE TABLE scenario_c_candidate_with_url AS
    SELECT
        c.*,
        u.SOURCEURL
    FROM scenario_c_tasking_candidate_core c
    LEFT JOIN gdelt_url_lookup u USING (GLOBALEVENTID)
    """)

    retention = con.sql("""
    SELECT 'broad' AS stage, COUNT(*) AS cnt FROM scenario_c_broad
    UNION ALL
    SELECT 'combat', COUNT(*) FROM scenario_c_combat
    UNION ALL
    SELECT 'candidate_all', COUNT(*) FROM scenario_c_tasking_candidate_all
    UNION ALL
    SELECT 'candidate_core', COUNT(*) FROM scenario_c_tasking_candidate_core
    UNION ALL
    SELECT 'candidate_holdout', COUNT(*) FROM scenario_c_tasking_candidate_holdout
    """).df()
    retention.to_csv(OUT / "filter_retention.csv", index=False)

    event_root_distribution = con.sql("""
    SELECT EventRootCode, COUNT(*) AS cnt
    FROM scenario_c_broad
    GROUP BY 1
    ORDER BY cnt DESC
    """).df()
    event_root_distribution.to_csv(OUT / "event_root_distribution.csv", index=False)

    event_code_top20 = con.sql("""
    SELECT EventCode, COUNT(*) AS cnt
    FROM scenario_c_broad
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
    FROM scenario_c_broad
    GROUP BY 1,2
    ORDER BY cnt DESC
    """).df()
    dyad_distribution.to_csv(OUT / "dyad_distribution.csv", index=False)

    core_hotspots = con.sql("""
    SELECT
        ActionGeo_FeatureID,
        ActionGeo_FullName,
        ActionGeo_Lat,
        ActionGeo_Long,
        COUNT(*) AS event_cnt,
        SUM(COALESCE(NumMentions,0)) AS mention_sum,
        AVG(COALESCE(GoldsteinScale,0)) AS avg_goldstein,
        AVG(COALESCE(AvgTone,0)) AS avg_tone
    FROM scenario_c_tasking_candidate_core
    GROUP BY 1,2,3,4
    ORDER BY event_cnt DESC, mention_sum DESC
    LIMIT 20
    """).df()
    core_hotspots.to_csv(OUT / "hotspots_candidate_core_top20.csv", index=False)

    holdout_hotspots = con.sql("""
    SELECT
        ActionGeo_FeatureID,
        ActionGeo_FullName,
        ActionGeo_Lat,
        ActionGeo_Long,
        COUNT(*) AS event_cnt,
        SUM(COALESCE(NumMentions,0)) AS mention_sum,
        AVG(COALESCE(GoldsteinScale,0)) AS avg_goldstein,
        AVG(COALESCE(AvgTone,0)) AS avg_tone
    FROM scenario_c_tasking_candidate_holdout
    GROUP BY 1,2,3,4
    ORDER BY event_cnt DESC, mention_sum DESC
    LIMIT 20
    """).df()
    holdout_hotspots.to_csv(OUT / "hotspots_candidate_holdout_top20.csv", index=False)

    yearly_counts = con.sql("""
    SELECT
        CAST(SUBSTR(CAST(SQLDATE AS VARCHAR), 1, 4) AS INTEGER) AS year,
        COUNT(*) AS event_cnt,
        SUM(COALESCE(NumMentions,0)) AS mention_sum,
        AVG(COALESCE(GoldsteinScale,0)) AS avg_goldstein
    FROM scenario_c_combat
    GROUP BY 1
    ORDER BY 1
    """).df()
    yearly_counts.to_csv(OUT / "yearly_counts.csv", index=False)

    gt = con.sql("""
    WITH ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY SQLDATE, ActionGeo_FeatureID, EventCode
                   ORDER BY NumMentions DESC
               ) AS rn
        FROM scenario_c_candidate_with_url
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

    gt = fix_swapped_gt_columns(gt)
    gt.to_csv(OUT / "ground_truth_candidates_dedup.csv", index=False)

    con.execute(f"COPY scenario_c_broad TO '{(OUT / 'scenario_c_broad.parquet').as_posix()}' (FORMAT parquet)")
    con.execute(f"COPY scenario_c_combat TO '{(OUT / 'scenario_c_combat.parquet').as_posix()}' (FORMAT parquet)")
    con.execute(f"COPY scenario_c_tasking_candidate_all TO '{(OUT / 'scenario_c_tasking_candidate_all.parquet').as_posix()}' (FORMAT parquet)")
    con.execute(f"COPY scenario_c_tasking_candidate_core TO '{(OUT / 'scenario_c_tasking_candidate_core.parquet').as_posix()}' (FORMAT parquet)")
    con.execute(f"COPY scenario_c_tasking_candidate_holdout TO '{(OUT / 'scenario_c_tasking_candidate_holdout.parquet').as_posix()}' (FORMAT parquet)")

# -----------------------------
# 공통 메모 저장
# -----------------------------
(OUT / "scenario_notes.txt").write_text("\n".join(notes), encoding="utf-8")

print(f"완료: {OUT}")
print("생성 파일을 확인하세요.")