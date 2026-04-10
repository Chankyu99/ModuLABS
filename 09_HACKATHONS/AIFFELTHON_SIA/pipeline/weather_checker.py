"""
Level 2a — 기상 판별기
────────────────────────────────
Open-Meteo API로 위성 통과 시점의 구름량과 주/야간 상태를 판별한다.
- API 키 불필요, 호출 제한 사실상 없음
- 1시간 간격 구름량 예보 (16일)
"""
from __future__ import annotations

import requests
from datetime import datetime, timezone
from functools import lru_cache

from astral import LocationInfo
from astral.sun import sun

from pipeline.config import CLOUD_THRESHOLD


# ──────────────────────────────────────────────
# Open-Meteo API
# ──────────────────────────────────────────────
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


@lru_cache(maxsize=64)
def _fetch_cloud_forecast(lat: float, lon: float) -> dict | None:
    """Open-Meteo에서 시간별 구름량 예보를 가져온다. 좌표별 캐싱."""
    # 좌표를 소수점 2자리로 반올림하여 인접 요청 병합
    lat_r = round(lat, 2)
    lon_r = round(lon, 2)

    try:
        resp = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat_r,
                "longitude": lon_r,
                "hourly": "cloud_cover",
                "timezone": "UTC",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        
        # O(1) 조회를 위해 HashMap(Dictionary) 사전 구축
        if "hourly" in data:
            times = data["hourly"]["time"]
            clouds = data["hourly"]["cloud_cover"]
            data["cloud_map"] = dict(zip(times, clouds))
            
        return data
    except requests.RequestException as e:
        print(f"  [WEATHER] ⚠️ Open-Meteo 요청 실패 ({lat_r}, {lon_r}): {e}")
        return None


def get_cloud_cover(lat: float, lon: float, pass_time_utc: str) -> dict:
    """특정 시각의 구름량 정보를 반환한다.

    Args:
        lat, lon: 좌표
        pass_time_utc: ISO 형식 UTC 시각 (예: "2026-04-09T14:23:00Z")

    Returns:
        {
            "cloud_cover_pct": int,
            "cloud_status": "clear" | "partial" | "overcast",
            "shootable_eo": bool,
        }
    """
    data = _fetch_cloud_forecast(lat, lon)

    if data is None or "cloud_map" not in data:
        return {
            "cloud_cover_pct": -1,
            "cloud_status": "unknown",
            "shootable_eo": False,
        }

    # pass_time을 "YYYY-MM-DDTHH:00" 형식으로 변환 (반올림을 통한 정각 매칭)
    pass_dt = datetime.fromisoformat(pass_time_utc.replace("Z", "+00:00"))
    
    # 30분 이상이면 다음 시간 정각으로 올림
    if pass_dt.minute >= 30:
        from datetime import timedelta
        pass_dt += timedelta(hours=1)
        
    pass_dt = pass_dt.replace(minute=0, second=0, microsecond=0)
    target_hour = pass_dt.strftime("%Y-%m-%dT%H:00")

    # Hash 맵에서 O(1) 룩업
    cloud_pct = data["cloud_map"].get(target_hour, -1)

    # 구름 상태 판정
    if cloud_pct < 0:
        status = "unknown"
        shootable = False
    elif cloud_pct <= 20:
        status = "clear"
        shootable = True
    elif cloud_pct <= CLOUD_THRESHOLD:
        status = "partial"
        shootable = True
    else:
        status = "overcast"
        shootable = False

    return {
        "cloud_cover_pct": cloud_pct,
        "cloud_status": status,
        "shootable_eo": shootable,
    }


def is_daylight(lat: float, lon: float, pass_time_utc: str) -> bool:
    """통과 시각이 주간(일출~일몰)인지 판별한다."""
    pass_dt = datetime.fromisoformat(pass_time_utc.replace("Z", "+00:00"))

    loc = LocationInfo(latitude=lat, longitude=lon)
    try:
        s = sun(loc.observer, date=pass_dt.date(), tzinfo=timezone.utc)
        return s["sunrise"] <= pass_dt <= s["sunset"]
    except Exception:
        # 극지방 등 일출/일몰 계산 불가 시 주간으로 간주
        return True


def check_weather(pass_event: dict) -> dict:
    """통과 이벤트에 구름량 + 주야간 정보를 추가한다.

    Args:
        pass_event: pass_predictor에서 반환된 통과 이벤트 dict

    Returns:
        원본 dict에 weather 필드가 추가된 새 dict
    """
    lat = pass_event["lat"]
    lon = pass_event["lon"]
    pass_time = pass_event["pass_time_utc"]
    sensor = pass_event["sensor_type"]

    cloud = get_cloud_cover(lat, lon, pass_time)
    daylight = is_daylight(lat, lon, pass_time)

    # SAR은 구름·야간 무관하게 항상 촬영 가능
    if sensor == "sar":
        shootable = True
    else:
        shootable = cloud["shootable_eo"] and daylight

    result = pass_event.copy()
    result.update({
        "cloud_cover_pct": cloud["cloud_cover_pct"],
        "cloud_status": cloud["cloud_status"],
        "daylight": daylight,
        "shootable": shootable,
    })

    return result


# ──────────────────────────────────────────────
# CLI 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="기상 판별기")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    cloud = get_cloud_cover(args.lat, args.lon, now_utc)
    daylight = is_daylight(args.lat, args.lon, now_utc)

    print(f"\n  ── 기상 판별 결과 ──")
    print(f"  좌표: ({args.lat}, {args.lon})")
    print(f"  시각: {now_utc}")
    print(f"  구름량: {cloud['cloud_cover_pct']}%  ({cloud['cloud_status']})")
    print(f"  주간 여부: {'☀️ 주간' if daylight else '🌙 야간'}")
    print(f"  EO 촬영 가능: {'✅' if cloud['shootable_eo'] and daylight else '❌'}")
