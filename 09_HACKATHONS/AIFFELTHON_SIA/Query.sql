-- Parquet로 저장
EXPORT DATA OPTIONS(
  uri='gs://gdelt-eda-0331/gdelt_eda_*.parquet',
  format='PARQUET',
  overwrite=true
) AS
SELECT
  GLOBALEVENTID, SQLDATE,
  Actor1Name, Actor1CountryCode,
  Actor2Name, Actor2CountryCode,
  IsRootEvent, EventCode, EventRootCode,
  QuadClass, GoldsteinScale,AvgTone,
  ActionGeo_FullName, ActionGeo_CountryCode,
  ActionGeo_Lat, ActionGeo_Long,
  ActionGeo_FeatureID, ActionGeo_Type, 
  NumMentions, NumSources, NumArticles,
  SOURCEURL
FROM `gdelt-bq.full.events`
WHERE SQLDATE BETWEEN 20130401 AND 20260331
  AND (
    Actor1CountryCode IN ('IRN', 'USA', 'ISR')
    OR Actor2CountryCode IN ('IRN', 'USA', 'ISR')
  )
  AND ActionGeo_Lat BETWEEN 12 AND 42
  AND ActionGeo_Long BETWEEN 25 AND 65