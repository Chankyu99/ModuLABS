"""
Ground truth 로더
─────────────────
- 기존 라벨형 GT(`date`, `city`, `manual_label`, `selection_decision`) 지원
- 신규 positive-only GT(`SQLDATE`, `ActionGeo_FullName`) 지원
- 여러 CSV를 하나로 병합하고, 날짜 불일치 파일을 리포트한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class GroundTruthLoadResult:
    dataframe: pd.DataFrame
    is_positive_only: bool
    file_summaries: list[dict]
    date_mismatches: list[dict]


def _expand_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        if any(ch in value for ch in "*?[]"):
            if path.is_absolute():
                paths.extend(sorted(path.parent.glob(path.name)))
            else:
                paths.extend(sorted(Path().glob(value)))
            continue
        if path.is_dir():
            paths.extend(sorted(path.glob("*.csv")))
            continue
        paths.append(path)
    deduped = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def load_ground_truth(values: list[str]) -> GroundTruthLoadResult:
    paths = _expand_paths(values)
    if not paths:
        raise FileNotFoundError("ground truth 파일이 없습니다.")

    frames = []
    file_summaries: list[dict] = []
    date_mismatches: list[dict] = []
    positive_only_flags = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"ground truth 파일이 없습니다: {path}")

        df = pd.read_csv(path)
        if {"SQLDATE", "ActionGeo_FullName"}.issubset(df.columns):
            normalized = pd.DataFrame(index=df.index)
            normalized["date"] = df["SQLDATE"].astype(str)
            normalized["city"] = df["ActionGeo_FullName"].astype(str).str.strip()
            normalized["source_file"] = path.name
            normalized["gt_schema"] = "positive_only"

            # 신규 GT 포맷에서 자주 쓰는 메타데이터는 가능한 한 그대로 보존한다.
            if "Lat" in df.columns:
                normalized["gt_lat"] = pd.to_numeric(df["Lat"], errors="coerce")
            if "Long" in df.columns:
                normalized["gt_lon"] = pd.to_numeric(df["Long"], errors="coerce")
            if "Source" in df.columns:
                normalized["gt_source"] = df["Source"].astype(str)
            if "Event_Description" in df.columns:
                normalized["gt_event_description"] = df["Event_Description"].astype(str)
            positive_only_flags.append(True)

            unique_sql_dates = sorted(normalized["date"].unique().tolist())
            file_summaries.append(
                {
                    "file": path.name,
                    "schema": "positive_only",
                    "rows": int(len(normalized)),
                    "dates": unique_sql_dates,
                    "cities": int(normalized["city"].nunique()),
                }
            )
            file_date = path.stem
            if len(unique_sql_dates) == 1 and unique_sql_dates[0] != file_date:
                date_mismatches.append(
                    {
                        "file": path.name,
                        "file_date": file_date,
                        "sql_date": unique_sql_dates[0],
                    }
                )

        elif {"date", "city"}.issubset(df.columns):
            normalized = df.copy()
            normalized["date"] = normalized["date"].astype(str)
            normalized["city"] = normalized["city"].astype(str).str.strip()
            normalized["source_file"] = path.name
            normalized["gt_schema"] = "labeled"
            positive_only_flags.append(False)
            file_summaries.append(
                {
                    "file": path.name,
                    "schema": "labeled",
                    "rows": int(len(normalized)),
                    "dates": sorted(normalized["date"].astype(str).unique().tolist()),
                    "cities": int(normalized["city"].astype(str).nunique()),
                }
            )
        else:
            raise ValueError(
                f"지원하지 않는 ground truth 형식입니다: {path.name} / columns={list(df.columns)}"
            )

        frames.append(normalized)

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = combined["date"].astype(str)
    combined["city"] = combined["city"].astype(str).str.strip()
    combined = combined.drop_duplicates(subset=["date", "city"]).reset_index(drop=True)

    # 모든 파일이 positive-only일 때만 positive-only mode로 간주한다.
    is_positive_only = bool(positive_only_flags) and all(positive_only_flags)
    return GroundTruthLoadResult(
        dataframe=combined,
        is_positive_only=is_positive_only,
        file_summaries=file_summaries,
        date_mismatches=date_mismatches,
    )
