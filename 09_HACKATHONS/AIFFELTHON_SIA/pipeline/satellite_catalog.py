"""
Level 2a — 위성 카탈로그 로더
────────────────────────────────
정적 기본 위성 목록과 eo-predictor 기반 별도 시나리오를 공통 형식으로 제공한다.
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline.config import SATELLITES as DEFAULT_SATELLITES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EO_PREDICTOR_SAT_DIR = PROJECT_ROOT / "eo-predictor" / "scripts" / "satellites"


def _default_spaceeye_entry() -> dict:
    for sat in DEFAULT_SATELLITES:
        if sat["name"] == "SpaceEye-T":
            return sat.copy()
    raise ValueError("기본 SATELLITES에서 SpaceEye-T를 찾을 수 없습니다.")


def _load_eo_constellation(filename: str) -> list[dict]:
    file_path = EO_PREDICTOR_SAT_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"EO Predictor 위성 정의 파일이 없습니다: {file_path}")

    constellation = json.loads(file_path.read_text(encoding="utf-8"))
    satellites = []
    for norad_id in constellation.get("norad_ids", []):
        satellites.append({
            "name": f"{constellation['constellation']}-{norad_id}",
            "norad_id": int(norad_id),
            "type": str(constellation["sensor_type"]).lower(),
            "swath_km": float(constellation["swath_km"]),
            "resolution_m": float(constellation["spatial_res_cm"]) / 100.0,
            "off_nadir_deg": constellation.get("off_nadir_deg"),
            "orbit": "SSO",
            "altitude_km": float(constellation["altitude_km"]),
            "priority": 10,
            "constellation": constellation["constellation"],
            "operator": constellation.get("operator", ""),
            "data_access": constellation.get("data_access", ""),
            "tasking": constellation.get("tasking"),
            "source": "eo-predictor",
        })
    return satellites


def load_satellite_catalog(scenario: str = "default") -> list[dict]:
    """실행 시나리오에 맞는 위성 목록을 반환한다."""
    if scenario == "default":
        return [sat.copy() for sat in DEFAULT_SATELLITES]

    if scenario == "tri-mix":
        satellites = []
        satellites.extend(_load_eo_constellation("iceye.json"))
        satellites.extend(_load_eo_constellation("planetscope.json"))
        spaceeye = _default_spaceeye_entry()
        spaceeye["priority"] = 1
        spaceeye["source"] = "custom-default"
        satellites.append(spaceeye)
        return satellites

    if scenario == "iceye-spaceeye":
        satellites = _load_eo_constellation("iceye.json")
        spaceeye = _default_spaceeye_entry()
        spaceeye["priority"] = 1
        spaceeye["source"] = "custom-default"
        satellites.append(spaceeye)
        return satellites

    if scenario == "planetscope-spaceeye":
        satellites = _load_eo_constellation("planetscope.json")
        spaceeye = _default_spaceeye_entry()
        spaceeye["priority"] = 1
        spaceeye["source"] = "custom-default"
        satellites.append(spaceeye)
        return satellites

    raise ValueError(f"지원하지 않는 위성 시나리오입니다: {scenario}")
