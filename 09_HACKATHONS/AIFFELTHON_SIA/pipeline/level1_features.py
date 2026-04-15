"""
Level 1 — ARIMA 친화형 갈등 지수/태스킹 점수 뼈대
────────────────────────────────────────────────────
- GDELT 원천 이벤트를 도시-일(city-day) feature 테이블로 집계
- EDA 기반으로 해석 가능한 서브 인덱스(Media / Event Severity / Hostility) 생성
- 도시별 robust scaling을 거쳐 ARIMA에 넣기 좋은 단일 갈등 지수 생성
- ARIMA residual z-score + 절대 수준(level z)를 결합한 tasking score 산출

의도:
- 기존 휴리스틱 conflict_index와 분리된 "ARIMA friendly" baseline을 만든다.
- Kalman / ARIMA 비교 실험에서 입력 지수 자체를 교체해볼 수 있는 뼈대를 제공한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.config import (
    CONFIRMED_CODES,
    MIN_HISTORY,
    MONITORED_COUNTRIES,
    get_risk_level,
)
from pipeline.level1_arima import rolling_arima_anomaly


HIGH_SEVERITY_ROOT_CODES = {18, 19, 20}


@dataclass(frozen=True)
class ArimaFriendlyWeights:
    """EDA 기반 초기 가중치 baseline.

    이 값은 현재 "해석 가능한 초기값"입니다.
    이후 EDA/ground truth로 튜닝하는 것이 전제입니다.
    """

    mi_mentions: float = 0.50
    mi_sources: float = 0.30
    mi_articles: float = 0.20

    es_high_severity_ratio: float = 0.60
    es_event_count: float = 0.40

    ht_goldstein: float = 0.60
    ht_tone: float = 0.40

    conflict_media: float = 0.40
    conflict_event: float = 0.35
    conflict_hostility: float = 0.25

    tasking_surprise: float = 0.70
    tasking_level: float = 0.30


def _safe_iqr(values: pd.Series) -> float:
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = float(q3 - q1)
    return iqr if iqr > 1e-8 else 1.0


def _robust_zscore_by_city(df: pd.DataFrame, column: str, out_col: str) -> pd.DataFrame:
    """도시별 median/IQR 기준 robust z-score 생성."""
    result = df.copy()

    def _transform(group: pd.DataFrame) -> pd.DataFrame:
        median = float(group[column].median())
        iqr = _safe_iqr(group[column])
        group[out_col] = (group[column] - median) / iqr
        return group

    return result.groupby("city", group_keys=False).apply(_transform).reset_index(drop=True)


def build_city_day_features(
    df: pd.DataFrame,
    min_sources: int = 2,
) -> pd.DataFrame:
    """GDELT 이벤트를 도시-일 단위 feature 테이블로 변환한다."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "city",
                "country_code",
                "lat",
                "lon",
                "event_count",
                "mentions_sum",
                "sources_sum",
                "articles_sum",
                "negative_tone_mean",
                "negative_goldstein_mean",
                "high_severity_ratio",
                "root_event_ratio",
            ]
        )

    filtered = df.copy()
    mask = (
        (filtered["Actor1CountryCode"].isin(MONITORED_COUNTRIES)
         | filtered["Actor2CountryCode"].isin(MONITORED_COUNTRIES))
        & pd.to_numeric(filtered["EventCode"], errors="coerce").isin(CONFIRMED_CODES)
        & (filtered["ActionGeo_Type"] == 4)
        & (filtered["NumSources"] >= min_sources)
    )
    filtered = filtered.loc[mask].copy()
    if filtered.empty:
        return pd.DataFrame()

    # 기존 Level 1과 같은 취지로 중복 기사 팽창을 완화
    if {"Actor1Name", "Actor2Name"}.issubset(filtered.columns):
        filtered["info_count"] = filtered[["Actor1Name", "Actor2Name"]].notna().sum(axis=1)
        filtered = filtered.sort_values(by="info_count", ascending=False)
        filtered = filtered.drop_duplicates(
            subset=["SQLDATE", "ActionGeo_FeatureID", "EventCode", "AvgTone", "NumSources"],
            keep="first",
        )

    filtered["date"] = filtered["SQLDATE"].astype(str).str[:8]
    filtered["city"] = filtered["ActionGeo_FullName"].astype(str).str.split(",").str[0].str.strip()
    filtered["country_code"] = filtered["ActionGeo_CountryCode"].fillna("")
    filtered["negative_tone"] = np.maximum(0.0, -pd.to_numeric(filtered["AvgTone"], errors="coerce").fillna(0.0))
    filtered["negative_goldstein"] = np.maximum(
        0.0,
        -pd.to_numeric(filtered["GoldsteinScale"], errors="coerce").fillna(0.0),
    )
    filtered["high_severity_flag"] = (
        pd.to_numeric(filtered["EventRootCode"], errors="coerce")
        .fillna(0)
        .astype(int)
        .isin(HIGH_SEVERITY_ROOT_CODES)
        .astype(int)
    )
    filtered["root_event_flag"] = pd.to_numeric(filtered["IsRootEvent"], errors="coerce").fillna(0).astype(int)

    city_day = (
        filtered.groupby(["date", "city"])
        .agg(
            country_code=("country_code", "first"),
            lat=("ActionGeo_Lat", "median"),
            lon=("ActionGeo_Long", "median"),
            event_count=("EventCode", "count"),
            mentions_sum=("NumMentions", "sum"),
            sources_sum=("NumSources", "sum"),
            articles_sum=("NumArticles", "sum"),
            negative_tone_mean=("negative_tone", "mean"),
            negative_goldstein_mean=("negative_goldstein", "mean"),
            high_severity_ratio=("high_severity_flag", "mean"),
            root_event_ratio=("root_event_flag", "mean"),
        )
        .reset_index()
    )

    # 날짜 continuity를 보장해 ARIMA 입력에 적합한 형태로 맞춘다.
    filled_parts = []
    for city, group in city_day.groupby("city"):
        city_dates = pd.date_range(
            start=pd.to_datetime(group["date"].min(), format="%Y%m%d"),
            end=pd.to_datetime(group["date"].max(), format="%Y%m%d"),
            freq="D",
        ).strftime("%Y%m%d")

        city_country = group["country_code"].dropna().iloc[0] if group["country_code"].notna().any() else ""
        city_lat = float(group["lat"].dropna().median()) if group["lat"].notna().any() else np.nan
        city_lon = float(group["lon"].dropna().median()) if group["lon"].notna().any() else np.nan

        group = group.set_index("date").reindex(city_dates)
        group["city"] = city
        group["country_code"] = group["country_code"].fillna(city_country)
        group["lat"] = group["lat"].fillna(city_lat)
        group["lon"] = group["lon"].fillna(city_lon)

        zero_fill_cols = [
            "event_count",
            "mentions_sum",
            "sources_sum",
            "articles_sum",
            "negative_tone_mean",
            "negative_goldstein_mean",
            "high_severity_ratio",
            "root_event_ratio",
        ]
        for col in zero_fill_cols:
            group[col] = group[col].fillna(0.0)

        group.index.name = "date"
        filled_parts.append(group.reset_index())

    result = pd.concat(filled_parts, ignore_index=True)
    return result.sort_values(["date", "city"]).reset_index(drop=True)


def compute_arima_friendly_conflict_index(
    city_day: pd.DataFrame,
    weights: ArimaFriendlyWeights = ArimaFriendlyWeights(),
) -> pd.DataFrame:
    """EDA 기반 갈등 지수 baseline을 계산한다."""
    if city_day.empty:
        return city_day.copy()

    result = city_day.copy()

    result["mi_raw"] = (
        weights.mi_mentions * np.log1p(result["mentions_sum"])
        + weights.mi_sources * np.log1p(result["sources_sum"])
        + weights.mi_articles * np.log1p(result["articles_sum"])
    )

    result["es_raw"] = (
        weights.es_high_severity_ratio * result["high_severity_ratio"]
        + weights.es_event_count * np.log1p(result["event_count"])
    )

    result["ht_raw"] = (
        weights.ht_goldstein * result["negative_goldstein_mean"]
        + weights.ht_tone * result["negative_tone_mean"]
    )

    result = _robust_zscore_by_city(result, "mi_raw", "mi_z")
    result = _robust_zscore_by_city(result, "es_raw", "es_z")
    result = _robust_zscore_by_city(result, "ht_raw", "ht_z")

    result["conflict_index_arima"] = (
        weights.conflict_media * result["mi_z"]
        + weights.conflict_event * result["es_z"]
        + weights.conflict_hostility * result["ht_z"]
    )

    result = _robust_zscore_by_city(result, "conflict_index_arima", "level_z")
    return result


def detect_tasking_with_arima(
    city_index: pd.DataFrame,
    min_history: int = MIN_HISTORY,
    transform: str = "log1p",
    weights: ArimaFriendlyWeights = ArimaFriendlyWeights(),
) -> pd.DataFrame:
    """ARIMA residual 기반 태스킹 점수를 계산한다."""
    if city_index.empty:
        return pd.DataFrame()

    results = []
    for city, group in city_index.groupby("city"):
        group = group.sort_values("date").reset_index(drop=True)
        if len(group) < min_history:
            continue

        arima_df = rolling_arima_anomaly(
            group["conflict_index_arima"],
            min_history=min_history,
            transform=transform,
        )

        group = group.copy()
        group["arima_forecast"] = arima_df["arima_forecast"]
        group["arima_residual"] = arima_df["arima_residual"]
        group["arima_z"] = arima_df["arima_z"].fillna(0.0)
        group["arima_order"] = arima_df["arima_order"]

        group["surprise_score"] = np.maximum(0.0, group["arima_z"])
        group["level_score"] = np.maximum(0.0, group["level_z"])
        group["tasking_score"] = (
            weights.tasking_surprise * group["surprise_score"]
            + weights.tasking_level * group["level_score"]
        )

        risk_info = group["tasking_score"].apply(get_risk_level)
        group["risk_level"] = risk_info.apply(lambda x: x["level"])
        group["risk_label"] = risk_info.apply(lambda x: x["label"])
        group["risk_emoji"] = risk_info.apply(lambda x: x["emoji"])
        group["risk_guide"] = risk_info.apply(lambda x: x["guide"])
        group["is_anomaly"] = group["risk_level"] >= 1

        results.append(group)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True).sort_values(
        ["date", "tasking_score"], ascending=[True, False]
    )


def run_level1_arima_friendly_pipeline(
    raw_df: pd.DataFrame,
    target_date: str | None = None,
    min_history: int = MIN_HISTORY,
    transform: str = "log1p",
    weights: ArimaFriendlyWeights = ArimaFriendlyWeights(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ARIMA 친화형 Level 1 pipeline 뼈대.

    Returns:
        city_day_features,
        city_index_features,
        detection_results
    """
    city_day = build_city_day_features(raw_df)
    city_index = compute_arima_friendly_conflict_index(city_day, weights=weights)
    detected = detect_tasking_with_arima(
        city_index,
        min_history=min_history,
        transform=transform,
        weights=weights,
    )

    if target_date is not None and not detected.empty:
        detected = detected[detected["date"].astype(str) == str(target_date)].copy()

    return city_day, city_index, detected
