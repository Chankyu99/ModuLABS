"""
Level 2a — 궤도 통과 예측기
────────────────────────────────
Skyfield SGP4 전파 엔진으로 ROI 상공 위성 통과(Overpass) 시각을 계산한다.
Phase 1: 직하(Nadir) Swath 기준 촬영 가능 판정
Phase 2: Off-Nadir 공개 위성은 확장 범위 반영
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

from skyfield.api import load, EarthSatellite, wgs84
from geopy.distance import geodesic

from pipeline.config import (
    SATELLITES, ROI_CITIES, MIN_ELEVATION_DEG, PREDICTION_HOURS,
)
from pipeline.tle_fetcher import load_all_tle


def _build_satellite(tle_entry: dict, ts) -> EarthSatellite:
    """TLE 데이터로 Skyfield EarthSatellite 객체를 생성한다."""
    return EarthSatellite(
        tle_entry["line1"],
        tle_entry["line2"],
        tle_entry["name"],
        ts,
    )


def predict_passes(
    tle_data: dict,
    cities: dict | None = None,
    hours: int | None = None,
) -> list[dict]:
    """모든 위성 × 모든 도시의 통과 이벤트를 예측한다.

    Args:
        tle_data: load_all_tle() 반환값
        cities: {city_name: {"lat": float, "lon": float}} (None이면 config ROI_CITIES)
        hours: 예측 범위(시간) (None이면 config PREDICTION_HOURS)

    Returns:
        통과 이벤트 리스트 (dict)
    """
    if cities is None:
        cities = ROI_CITIES
    if hours is None:
        hours = PREDICTION_HOURS

    ts = load.timescale()
    t0 = ts.now()
    t1 = ts.from_datetime(t0.utc_datetime() + timedelta(hours=hours))

    passes = []

    for norad_id_str, tle_entry in tle_data.items():
        meta = tle_entry["meta"]
        sat_name = meta["display_name"]
        swath_km = meta["swath_km"]
        off_nadir_deg = meta["off_nadir_deg"]
        altitude_km = meta["altitude_km"]
        priority = meta["priority"]

        try:
            satellite = _build_satellite(tle_entry, ts)
        except Exception as e:
            print(f"  [PASS] ⚠️ {sat_name} 위성 객체 생성 실패: {e}")
            continue

        for city_name, coord in cities.items():
            observer = wgs84.latlon(coord["lat"], coord["lon"])

            try:
                t_events, events = satellite.find_events(
                    observer, t0, t1, altitude_degrees=MIN_ELEVATION_DEG
                )
            except Exception:
                continue

            # find_events는 rise(0), culminate(1), set(2) 이벤트를 반환
            # culminate 이벤트만 추출하여 최대 앙각과 통과 시각을 구함
            i = 0
            while i < len(events):
                # culminate 이벤트 찾기
                if events[i] == 1:
                    t_culm = t_events[i]
                    culm_dt = t_culm.utc_datetime()

                    # 최대 앙각 계산
                    diff = satellite - observer
                    topocentric = diff.at(t_culm)
                    alt_deg, _, _ = topocentric.altaz()
                    max_elev = alt_deg.degrees

                    # 직하점(subpoint) 계산 → ROI까지 거리
                    geocentric = satellite.at(t_culm)
                    subpoint = wgs84.subpoint(geocentric)
                    sub_lat = subpoint.latitude.degrees
                    sub_lon = subpoint.longitude.degrees

                    distance_km = geodesic(
                        (sub_lat, sub_lon),
                        (coord["lat"], coord["lon"])
                    ).km

                    # Phase 1: Nadir Swath 범위 판정
                    nadir_range = swath_km / 2.0
                    within_nadir = distance_km <= nadir_range

                    # Phase 2: Off-Nadir 확장 범위 판정
                    within_offnadir = False
                    extended_range_km = nadir_range
                    if off_nadir_deg is not None and off_nadir_deg > 0:
                        extended_range_km = altitude_km * math.tan(
                            math.radians(off_nadir_deg)
                        )
                        within_offnadir = distance_km <= extended_range_km

                    within_swath = within_nadir or within_offnadir

                    passes.append({
                        "satellite": sat_name,
                        "norad_id": int(norad_id_str),
                        "sensor_type": meta["type"],
                        "resolution_m": meta["resolution_m"],
                        "priority": priority,
                        "city": city_name,
                        "lat": coord["lat"],
                        "lon": coord["lon"],
                        "pass_time_utc": culm_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "max_elevation_deg": round(max_elev, 1),
                        "subpoint_lat": round(sub_lat, 4),
                        "subpoint_lon": round(sub_lon, 4),
                        "distance_km": round(distance_km, 1),
                        "nadir_range_km": round(nadir_range, 1),
                        "extended_range_km": round(extended_range_km, 1),
                        "within_swath": within_swath,
                        "swath_phase": "nadir" if within_nadir else (
                            "off-nadir" if within_offnadir else "out-of-range"
                        ),
                    })

                i += 1

    # 시간순 정렬
    passes.sort(key=lambda x: x["pass_time_utc"])
    return passes


def filter_shootable(passes: list[dict]) -> list[dict]:
    """Swath 범위 내 통과만 필터링한다."""
    return [p for p in passes if p["within_swath"]]


# ──────────────────────────────────────────────
# CLI 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="위성 통과 예측기")
    parser.add_argument("--city", type=str, help="특정 도시만 예측")
    parser.add_argument("--hours", type=int, default=48, help="예측 범위(시간)")
    args = parser.parse_args()

    tle_data = load_all_tle()
    cities = None
    if args.city:
        if args.city in ROI_CITIES:
            cities = {args.city: ROI_CITIES[args.city]}
        else:
            print(f"도시 '{args.city}'을(를) 찾을 수 없습니다.")
            exit(1)

    print(f"\n  ── 위성 통과 예측 ({args.hours}시간) ──")
    all_passes = predict_passes(tle_data, cities=cities, hours=args.hours)
    shootable = filter_shootable(all_passes)

    print(f"\n  전체 통과: {len(all_passes)}건 | 촬영 가능: {len(shootable)}건\n")

    for p in shootable[:20]:
        print(
            f"  🛰️ {p['satellite']:12s} | {p['city']:15s} | "
            f"{p['pass_time_utc']} | 앙각 {p['max_elevation_deg']:5.1f}° | "
            f"거리 {p['distance_km']:6.1f}km | {p['swath_phase']}"
        )

    if len(shootable) > 20:
        print(f"  ... 외 {len(shootable)-20}건")
