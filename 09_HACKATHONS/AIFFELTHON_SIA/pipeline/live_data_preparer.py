"""
Fetch and prepare live GDELT inputs for the integrated pipeline.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from pipeline.archive.config import (
    ACTION_GEO_ALLOWED_COUNTRIES,
    CONFIRMED_CODES,
    MONITORED_COUNTRIES,
)
from pipeline.integrated_pipeline import resolve_main_data_path, resolve_url_data_path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = PROJECT_ROOT / "data" / "daily"
GENERATED_DIR = PROJECT_ROOT / "data" / "generated"

GDELT_BASE_URL = "http://data.gdeltproject.org/events/"

GDELT_COLUMNS = [
    "GLOBALEVENTID",
    "SQLDATE",
    "MonthYear",
    "Year",
    "FractionDate",
    "Actor1Code",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor1KnownGroupCode",
    "Actor1EthnicCode",
    "Actor1Religion1Code",
    "Actor1Religion2Code",
    "Actor1Type1Code",
    "Actor1Type2Code",
    "Actor1Type3Code",
    "Actor2Code",
    "Actor2Name",
    "Actor2CountryCode",
    "Actor2KnownGroupCode",
    "Actor2EthnicCode",
    "Actor2Religion1Code",
    "Actor2Religion2Code",
    "Actor2Type1Code",
    "Actor2Type2Code",
    "Actor2Type3Code",
    "IsRootEvent",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "Actor1Geo_Type",
    "Actor1Geo_FullName",
    "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code",
    "Actor1Geo_Lat",
    "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type",
    "Actor2Geo_FullName",
    "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code",
    "Actor2Geo_Lat",
    "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type",
    "ActionGeo_FullName",
    "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED",
    "SOURCEURL",
]


@dataclass(frozen=True)
class PreparedLiveInputs:
    main_path: Path
    url_path: Path
    fetched_dates: list[str]
    base_max_date: str | None


def _parse_yyyymmdd(value: str) -> datetime:
    return datetime.strptime(str(value), "%Y%m%d")


def compute_fetch_dates(base_max_date: str | None, target_date: str) -> list[str]:
    if base_max_date is None or str(base_max_date) >= str(target_date):
        return []

    start = _parse_yyyymmdd(base_max_date) + timedelta(days=1)
    end = _parse_yyyymmdd(target_date)
    return [
        (start + timedelta(days=offset)).strftime("%Y%m%d")
        for offset in range((end - start).days + 1)
    ]


def _detect_max_sql_date(main_path: Path) -> str | None:
    df = pd.read_parquet(main_path, columns=["SQLDATE"])
    if df.empty:
        return None
    return df["SQLDATE"].astype(str).str[:8].max()


def _cast_numeric_column(frame: pd.DataFrame, column: str, dtype: str) -> None:
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if dtype.startswith(("int", "uint")):
        frame[column] = frame[column].fillna(0).astype(dtype)
    else:
        frame[column] = frame[column].astype(dtype)


def cast_to_schema(frame: pd.DataFrame, schema_frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in schema_frame.columns:
        if column not in normalized.columns:
            normalized[column] = pd.NA

    normalized = normalized[list(schema_frame.columns)]

    for column, dtype in schema_frame.dtypes.items():
        if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_unsigned_integer_dtype(dtype):
            _cast_numeric_column(normalized, column, str(dtype))
        elif pd.api.types.is_float_dtype(dtype):
            _cast_numeric_column(normalized, column, str(dtype))
        elif pd.api.types.is_bool_dtype(dtype):
            normalized[column] = normalized[column].fillna(False).astype(bool)
        else:
            normalized[column] = normalized[column].astype("string")

    return normalized


def _cast_url_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["GLOBALEVENTID", "SOURCEURL"])

    normalized = frame.copy()
    normalized["GLOBALEVENTID"] = pd.to_numeric(normalized["GLOBALEVENTID"], errors="coerce").fillna(0).astype("uint32")
    normalized["SOURCEURL"] = normalized["SOURCEURL"].astype("string")
    normalized = normalized.dropna(subset=["SOURCEURL"])
    normalized = normalized[normalized["SOURCEURL"].astype(str).str.len() > 0]
    normalized = normalized.drop_duplicates(subset=["GLOBALEVENTID"])
    return normalized


def fetch_and_filter_gdelt_day(target_date: str) -> pd.DataFrame:
    url = f"{GDELT_BASE_URL}{target_date}.export.CSV.zip"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            frame = pd.read_csv(
                handle,
                sep="\t",
                header=None,
                names=GDELT_COLUMNS,
                dtype=str,
                on_bad_lines="skip",
            )

    for column in [
        "GLOBALEVENTID",
        "SQLDATE",
        "EventCode",
        "ActionGeo_Type",
        "NumMentions",
        "NumSources",
        "NumArticles",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in ["AvgTone", "GoldsteinScale", "ActionGeo_Lat", "ActionGeo_Long"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    mask = (
        (frame["Actor1CountryCode"].isin(MONITORED_COUNTRIES) | frame["Actor2CountryCode"].isin(MONITORED_COUNTRIES))
        & frame["EventCode"].fillna(0).astype(int).isin(CONFIRMED_CODES)
        & (frame["ActionGeo_Type"].fillna(0).astype(int) == 4)
        & frame["ActionGeo_CountryCode"].isin(ACTION_GEO_ALLOWED_COUNTRIES)
    )

    filtered = frame.loc[mask].copy()
    filtered["ActionGeo_FullName"] = (
        filtered["ActionGeo_FullName"].astype(str).str.split(",").str[0].str.strip()
    )
    return filtered


def prepare_live_prediction_inputs(
    target_date: str,
    main_path: str | None = None,
    url_path: str | None = None,
) -> PreparedLiveInputs:
    main_data_path = Path(main_path) if main_path else resolve_main_data_path()
    url_data_path = Path(url_path) if url_path else resolve_url_data_path()

    base_max_date = _detect_max_sql_date(main_data_path)
    fetch_dates = compute_fetch_dates(base_max_date, target_date)
    if not fetch_dates:
        return PreparedLiveInputs(
            main_path=main_data_path,
            url_path=url_data_path,
            fetched_dates=[],
            base_max_date=base_max_date,
        )

    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    base_main = pd.read_parquet(main_data_path)
    fetched_frames: list[pd.DataFrame] = []
    fetched_url_frames: list[pd.DataFrame] = []

    for date_str in fetch_dates:
        filtered = fetch_and_filter_gdelt_day(date_str)
        normalized = cast_to_schema(filtered, base_main)
        normalized.to_parquet(DAILY_DIR / f"{date_str}.parquet", index=False)
        fetched_frames.append(normalized)
        if "SOURCEURL" in filtered.columns:
            fetched_url_frames.append(filtered[["GLOBALEVENTID", "SOURCEURL"]].copy())

    merged_main = pd.concat([base_main] + fetched_frames, ignore_index=True)
    merged_main = merged_main.drop_duplicates(subset=["GLOBALEVENTID"])
    merged_main = cast_to_schema(merged_main, base_main)

    merged_main_path = GENERATED_DIR / f"gdelt_main_{target_date}.parquet"
    merged_main.to_parquet(merged_main_path, index=False)

    base_url = pd.read_parquet(url_data_path)
    base_url = _cast_url_frame(base_url[["GLOBALEVENTID", "SOURCEURL"]].copy())
    merged_url = pd.concat([base_url] + [_cast_url_frame(frame) for frame in fetched_url_frames], ignore_index=True)
    merged_url = merged_url.drop_duplicates(subset=["GLOBALEVENTID"])
    merged_url_path = GENERATED_DIR / f"gdelt_url_{target_date}.parquet"
    merged_url.to_parquet(merged_url_path, index=False)

    return PreparedLiveInputs(
        main_path=merged_main_path,
        url_path=merged_url_path,
        fetched_dates=fetch_dates,
        base_max_date=base_max_date,
    )
