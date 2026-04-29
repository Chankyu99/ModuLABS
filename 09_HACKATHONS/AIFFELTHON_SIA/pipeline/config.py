"""
Shared pipeline config for Level2a and integrated runners.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
TLE_CACHE_DIR = PROJECT_ROOT / "data" / "tle"
OPERATIONAL_SATELLITE_SCENARIO = "coverage"


SATELLITES = [
    {
        "name": "SpaceEye-T",
        "norad_id": 63229,
        "type": "optical",
        "swath_km": 12,
        "resolution_m": 0.25,
        "off_nadir_deg": 45,
        "orbit": "SSO",
        "altitude_km": 510,
        "priority": 1,
    },
    {
        "name": "KOMPSAT-7",
        "norad_id": 66820,
        "type": "optical",
        "swath_km": 15,
        "resolution_m": 0.30,
        "off_nadir_deg": 30,
        "orbit": "SSO",
        "altitude_km": 570,
        "priority": 2,
    },
    {
        "name": "SkySat-C12",
        "norad_id": 43797,
        "type": "optical",
        "swath_km": 5.9,
        "resolution_m": 0.50,
        "off_nadir_deg": 30,
        "orbit": "SSO",
        "altitude_km": 500,
        "priority": 3,
    },
    {
        "name": "Sentinel-2A",
        "norad_id": 40697,
        "type": "optical",
        "swath_km": 290,
        "resolution_m": 10.0,
        "off_nadir_deg": None,
        "orbit": "SSO",
        "altitude_km": 786,
        "priority": 4,
    },
    {
        "name": "ICEYE-X2",
        "norad_id": 43800,
        "type": "sar",
        "swath_km": 30,
        "resolution_m": 1.0,
        "off_nadir_deg": 35,
        "orbit": "SSO",
        "altitude_km": 570,
        "priority": 5,
    },
]


CLOUD_THRESHOLD = 50
PREDICTION_HOURS = 168
MIN_ELEVATION_DEG = 20.0
MAX_OPERATIONAL_OFF_NADIR_DEG = 30.0


ROI_CITIES = {
    "Isfahan": {"lat": 32.6546, "lon": 51.6680},
    "Natanz": {"lat": 33.5130, "lon": 51.9220},
    "Bushehr": {"lat": 28.9684, "lon": 50.8385},
    "Tehran": {"lat": 35.6892, "lon": 51.3890},
    "Tabriz": {"lat": 38.0800, "lon": 46.2919},
    "Kharg Island": {"lat": 29.2333, "lon": 50.3167},
    "Dimona": {"lat": 31.0700, "lon": 35.2100},
    "Beirut": {"lat": 33.8938, "lon": 35.5018},
    "Baghdad": {"lat": 33.3152, "lon": 44.3661},
    "Gaza": {"lat": 31.5000, "lon": 34.4667},
    "Tel Aviv": {"lat": 32.0853, "lon": 34.7818},
    "Minab": {"lat": 27.1064, "lon": 57.0850},
    "Ras Laffan": {"lat": 25.9300, "lon": 51.5300},
    "Fujairah": {"lat": 25.1288, "lon": 56.3265},
    "Dubai": {"lat": 25.2048, "lon": 55.2708},
}
