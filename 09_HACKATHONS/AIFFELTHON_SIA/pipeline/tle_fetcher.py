"""
Level 2a — TLE 수집기
────────────────────────────
CelesTrak GP API에서 위성별 최신 TLE를 수집하여 로컬에 캐싱한다.
"""
from __future__ import annotations

import requests
import json
from datetime import datetime, timedelta
from pathlib import Path

from pipeline.config import SATELLITES, TLE_CACHE_DIR


# ──────────────────────────────────────────────
# CelesTrak API 설정
# ──────────────────────────────────────────────
CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"


def fetch_tle(norad_id: int) -> tuple[str, str, str] | None:
    """CelesTrak에서 단일 위성의 TLE를 가져온다.

    Returns:
        (name, line1, line2) 튜플 또는 실패 시 None
    """
    try:
        resp = requests.get(
            CELESTRAK_GP_URL,
            params={"CATNR": norad_id, "FORMAT": "TLE"},
            timeout=15,
        )
        resp.raise_for_status()

        lines = [l.strip() for l in resp.text.strip().split("\n") if l.strip()]
        if len(lines) < 3:
            print(f"  [TLE] NORAD {norad_id}: 데이터 불충분 ({len(lines)}줄)")
            return None

        name, line1, line2 = lines[0], lines[1], lines[2]
        return name, line1, line2

    except requests.RequestException as e:
        print(f"  [TLE] NORAD {norad_id}: 요청 실패 - {e}")
        return None


def _cache_path(date_str: str) -> Path:
    """날짜별 TLE 캐시 파일 경로."""
    TLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return TLE_CACHE_DIR / f"tle_{date_str}.json"


def load_all_tle(force_refresh: bool = False) -> dict:
    """모든 위성의 TLE를 수집하고 캐시에 저장한다.

    Returns:
        {norad_id: {"name": str, "line1": str, "line2": str, "meta": dict}, ...}
    """
    today = datetime.utcnow().strftime("%Y%m%d")
    cache_file = _cache_path(today)

    # 캐시가 있고 강제 갱신이 아니면 캐시 반환
    if cache_file.exists() and not force_refresh:
        print(f"  [TLE] 캐시 로드: {cache_file.name}")
        with open(cache_file, "r") as f:
            return json.load(f)

    print(f"  [TLE] CelesTrak에서 {len(SATELLITES)}개 위성의 TLE 수집 중...")
    result = {}

    for sat in SATELLITES:
        norad_id = sat["norad_id"]
        tle = fetch_tle(norad_id)

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
    with open(cache_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  [TLE] 캐시 저장: {cache_file.name} ({len(result)}개 위성)")

    return result


# ──────────────────────────────────────────────
# CLI 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CelesTrak TLE 수집기")
    parser.add_argument("--satellite", type=str, help="특정 위성만 수집")
    parser.add_argument("--refresh", action="store_true", help="캐시 무시, 재수집")
    args = parser.parse_args()

    if args.satellite:
        # 특정 위성만 조회
        sat_info = next(
            (s for s in SATELLITES if s["name"].lower() == args.satellite.lower()),
            None,
        )
        if sat_info:
            tle = fetch_tle(sat_info["norad_id"])
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
        data = load_all_tle(force_refresh=args.refresh)
        print(f"\n총 {len(data)}개 위성 TLE 수집 완료.")
