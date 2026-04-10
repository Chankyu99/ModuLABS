"""
Level 2a — TLE 수집기
────────────────────────────
CelesTrak GP API에서 위성별 최신 TLE를 수집하여 로컬에 캐싱한다.
"""
from __future__ import annotations

import requests
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline.config import SATELLITES, TLE_CACHE_DIR


# ──────────────────────────────────────────────
# CelesTrak API 설정
# ──────────────────────────────────────────────
CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"


def fetch_tle(norad_id: int, session: requests.Session | None = None) -> tuple[int, tuple[str, str, str] | None]:
    """CelesTrak에서 단일 위성의 TLE를 가져온다.

    Returns:
        (norad_id, (name, line1, line2) 또는 실패 시 None)
    """
    try:
        req_func = session.get if session else requests.get
        resp = req_func(
            CELESTRAK_GP_URL,
            params={"CATNR": norad_id, "FORMAT": "TLE"},
            timeout=15,
        )
        resp.raise_for_status()

        lines = [l.strip() for l in resp.text.strip().split("\n") if l.strip()]
        if len(lines) < 3:
            print(f"  [TLE] NORAD {norad_id}: 데이터 불충분 ({len(lines)}줄)")
            return norad_id, None

        name, line1, line2 = lines[0], lines[1], lines[2]
        return norad_id, (name, line1, line2)

    except requests.RequestException as e:
        print(f"  [TLE] NORAD {norad_id}: 요청 실패 - {e}")
        return norad_id, None


def _cache_path(date_str: str) -> Path:
    """날짜별 TLE 캐시 파일 경로."""
    TLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return TLE_CACHE_DIR / f"tle_{date_str}.json"


def _normalize_reference_date(reference_date: str | None = None) -> str:
    """YYYYMMDD 문자열을 정규화한다. None이면 오늘 UTC 날짜를 사용한다."""
    if reference_date:
        return str(reference_date)
    return datetime.utcnow().strftime("%Y%m%d")


def list_cached_tle_dates() -> list[str]:
    """로컬에 저장된 TLE 캐시 날짜 목록을 반환한다."""
    TLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dates = []
    for path in TLE_CACHE_DIR.glob("tle_*.json"):
        date_str = path.stem.replace("tle_", "", 1)
        if len(date_str) == 8 and date_str.isdigit():
            dates.append(date_str)
    return sorted(dates)


def resolve_tle_cache_date(reference_date: str | None = None, mode: str = "operational") -> str | None:
    """요청 날짜와 모드에 맞는 TLE 캐시 기준 날짜를 결정한다."""
    requested_date = _normalize_reference_date(reference_date)

    if mode != "backtest":
        return requested_date

    cached_dates = list_cached_tle_dates()
    if requested_date in cached_dates:
        return requested_date

    previous_dates = [d for d in cached_dates if d <= requested_date]
    if previous_dates:
        return previous_dates[-1]

    return None


def load_all_tle(
    force_refresh: bool = False,
    reference_date: str | None = None,
    mode: str = "operational",
    return_info: bool = False,
) -> dict | tuple[dict, dict]:
    """모든 위성의 TLE를 수집하고 캐시에 저장한다.

    Returns:
        {norad_id: {"name": str, "line1": str, "line2": str, "meta": dict}, ...}
    """
    requested_date = _normalize_reference_date(reference_date)
    resolved_date = resolve_tle_cache_date(requested_date, mode=mode)
    info = {
        "mode": mode,
        "requested_date": requested_date,
        "resolved_date": resolved_date,
        "source": "unknown",
    }

    if mode == "backtest":
        if force_refresh:
            print("  [TLE] 백테스트 모드에서는 과거 TLE 재수집 대신 캐시만 사용합니다.")

        if resolved_date is None:
            print(
                f"  [TLE] ❌ {requested_date} 이전 기준의 과거 TLE 캐시가 없습니다. "
                "data/tle/tle_YYYYMMDD.json 스냅샷이 필요합니다."
            )
            info["source"] = "missing-historical-cache"
            return ({}, info) if return_info else {}

        cache_file = _cache_path(resolved_date)
        source_label = "exact" if resolved_date == requested_date else "previous"
        print(
            f"  [TLE] 백테스트 캐시 로드: {cache_file.name} "
            f"(요청 {requested_date}, 사용 {resolved_date}, {source_label})"
        )
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        info["source"] = "historical-cache" if source_label == "exact" else "historical-cache-fallback"
        return (data, info) if return_info else data

    cache_file = _cache_path(requested_date)

    # 캐시가 있고 강제 갱신이 아니면 캐시 반환
    if cache_file.exists() and not force_refresh:
        print(f"  [TLE] 캐시 로드: {cache_file.name}")
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        info["source"] = "cache"
        return (data, info) if return_info else data

    print(f"  [TLE] CelesTrak에서 {len(SATELLITES)}개 위성의 TLE 수집 중... (병렬 처리)")
    result = {}

    with requests.Session() as session:
        # 워커 개수는 위성 개수만큼, 단 최대 10개로 제한
        max_workers = min(10, len(SATELLITES))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 퓨처 객체 매핑
            futures = {
                executor.submit(fetch_tle, sat["norad_id"], session): sat
                for sat in SATELLITES
            }
            
            for future in as_completed(futures):
                sat = futures[future]
                norad_id, tle = future.result()

                if tle is None:
                    print(f"  [TLE] ⚠️ {sat['name']} (NORAD {norad_id}) 수집 실패, 스킵")
                    continue

                name, line1, line2 = tle
                result[str(norad_id)] = {
                    "name": name,
                    "line1": line1,
                    "line2": line2,
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
                print(f"  [TLE] ✅ {sat['name']} ({name.strip()}) 수집 완료")

    # 캐시 저장
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  [TLE] 캐시 저장: {cache_file.name} ({len(result)}개 위성)")
    info["source"] = "fetch"

    return (result, info) if return_info else result


# ──────────────────────────────────────────────
# CLI 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CelesTrak TLE 수집기")
    parser.add_argument("--satellite", type=str, help="특정 위성만 수집")
    parser.add_argument("--refresh", action="store_true", help="캐시 무시, 재수집")
    parser.add_argument("--date", type=str, help="캐시 기준 날짜 (YYYYMMDD)")
    parser.add_argument("--mode", choices=["operational", "backtest"], default="operational", help="TLE 사용 모드")
    args = parser.parse_args()

    if args.satellite:
        # 특정 위성만 조회
        sat_info = next(
            (s for s in SATELLITES if s["name"].lower() == args.satellite.lower()),
            None,
        )
        if sat_info:
            _, tle = fetch_tle(sat_info["norad_id"])
            if tle:
                print(f"\n{'='*50}")
                print(f"위성: {sat_info['name']} (NORAD {sat_info['norad_id']})")
                print(f"{'='*50}")
                print(tle[0])
                print(tle[1])
                print(tle[2])
        else:
            print(f"위성 '{args.satellite}'을(를) 찾을 수 없습니다.")
    else:
        data = load_all_tle(
            force_refresh=args.refresh,
            reference_date=args.date,
            mode=args.mode,
        )
        print(f"\n총 {len(data)}개 위성 TLE 수집 완료.")
