#!/usr/bin/env python3
"""
Build backtest TLE cache files from pasted historical TLE text.

Usage:
  python3 build_historical_tle_cache.py --input data/tle/history_default.txt
  python3 build_historical_tle_cache.py --input data/tle/history_default.txt --dates 20260228 20260302
  python3 build_historical_tle_cache.py --input data/tle/history_default.txt --scenario default
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pipeline.config import TLE_CACHE_DIR
from pipeline.Level2a.satellite_catalog import load_satellite_catalog


PROJECT_ROOT = Path(__file__).resolve().parent
GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "ground_truth"
UTC = timezone.utc


@dataclass(frozen=True)
class TleRecord:
    norad_id: int
    line1: str
    line2: str
    epoch_utc: datetime
    name: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Historical TLE text -> Level2a cache JSON")
    parser.add_argument("--input", required=True, help="Path to raw pasted TLE text file")
    parser.add_argument("--dates", nargs="*", help="Target cache dates (YYYYMMDD). Default: all ground-truth dates")
    parser.add_argument("--scenario", default="default", help="Satellite catalog scenario")
    parser.add_argument("--allow-partial", action="store_true", help="Write cache even if some NORADs are missing")
    return parser.parse_args()


def load_target_dates(dates: list[str] | None) -> list[str]:
    if dates:
        return sorted(set(str(date) for date in dates))
    gt_dates = [
        path.stem
        for path in sorted(GROUND_TRUTH_DIR.glob("*.csv"))
        if path.stem.isdigit() and len(path.stem) == 8
    ]
    if not gt_dates:
        raise FileNotFoundError("No ground-truth CSV dates found. Pass --dates explicitly.")
    return gt_dates


def parse_tle_epoch(line1: str) -> datetime:
    parts = line1.split()
    if len(parts) < 4:
        raise ValueError(f"Invalid TLE line1: {line1}")
    epoch_token = parts[3]
    year_short = int(epoch_token[:2])
    year = 2000 + year_short if year_short < 57 else 1900 + year_short
    day_of_year = float(epoch_token[2:])
    day_index = int(day_of_year)
    day_fraction = day_of_year - day_index
    start = datetime(year, 1, 1, tzinfo=UTC)
    return start + timedelta(days=day_index - 1, seconds=day_fraction * 86400)


def parse_historical_tle_text(raw_text: str) -> list[TleRecord]:
    records: list[TleRecord] = []
    current_name: str | None = None
    pending_line1: str | None = None
    seen_pairs: set[tuple[int, str, str]] = set()

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("1 "):
            pending_line1 = line
            continue

        if line.startswith("2 "):
            if pending_line1 is None:
                continue
            norad_token = pending_line1.split()[1]
            norad_id = int("".join(ch for ch in norad_token if ch.isdigit())[:5])
            pair_key = (norad_id, pending_line1, line)
            if pair_key not in seen_pairs:
                records.append(
                    TleRecord(
                        norad_id=norad_id,
                        line1=pending_line1,
                        line2=line,
                        epoch_utc=parse_tle_epoch(pending_line1),
                        name=current_name,
                    )
                )
                seen_pairs.add(pair_key)
            pending_line1 = None
            current_name = None
            continue

        current_name = line

    return records


def build_cache_payload(
    records: list[TleRecord],
    target_date: str,
    scenario: str,
    allow_partial: bool = False,
) -> dict:
    satellites = load_satellite_catalog(scenario)
    satellite_by_norad = {int(sat["norad_id"]): sat for sat in satellites}
    expected_norads = sorted(satellite_by_norad)

    cutoff = datetime.strptime(target_date, "%Y%m%d").replace(tzinfo=UTC) + timedelta(days=1)
    latest_by_norad: dict[int, TleRecord] = {}

    for record in records:
        if record.norad_id not in satellite_by_norad:
            continue
        if record.epoch_utc >= cutoff:
            continue
        existing = latest_by_norad.get(record.norad_id)
        if existing is None or record.epoch_utc > existing.epoch_utc:
            latest_by_norad[record.norad_id] = record

    missing = [norad for norad in expected_norads if norad not in latest_by_norad]
    if missing and not allow_partial:
        missing_names = [satellite_by_norad[norad]["name"] for norad in missing]
        raise ValueError(
            f"{target_date}: missing TLE records for NORADs {missing} ({', '.join(missing_names)})"
        )

    payload = {}
    for norad_id in expected_norads:
        record = latest_by_norad.get(norad_id)
        if record is None:
            continue
        sat = satellite_by_norad[norad_id]
        payload[str(norad_id)] = {
            "name": record.name or sat["name"],
            "line1": record.line1,
            "line2": record.line2,
            "meta": {
                "display_name": sat["name"],
                "type": sat["type"],
                "swath_km": sat["swath_km"],
                "resolution_m": sat["resolution_m"],
                "off_nadir_deg": sat["off_nadir_deg"],
                "altitude_km": sat["altitude_km"],
                "priority": sat["priority"],
            },
        }

    return payload


def cache_output_path(target_date: str, scenario: str) -> Path:
    suffix = "" if scenario == "default" else f"_{scenario}"
    return TLE_CACHE_DIR / f"tle_{target_date}{suffix}.json"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    raw_text = input_path.read_text(encoding="utf-8")
    records = parse_historical_tle_text(raw_text)

    if not records:
        raise SystemExit("[ERROR] No TLE line pairs found in input text.")

    dates = load_target_dates(args.dates)
    TLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[TLE] parsed records: {len(records)}")
    print(f"[TLE] target dates: {', '.join(dates)}")
    print(f"[TLE] scenario: {args.scenario}")

    for target_date in dates:
        payload = build_cache_payload(
            records,
            target_date=target_date,
            scenario=args.scenario,
            allow_partial=args.allow_partial,
        )
        out_path = cache_output_path(target_date, args.scenario)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[TLE] wrote {out_path.name} ({len(payload)} satellites)")


if __name__ == "__main__":
    main()
