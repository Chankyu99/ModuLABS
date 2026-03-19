"""
spec_parser.py
==============
OCR 텍스트에서 항공 보안 규정 관련 핵심 수치를 추출하는 모듈.
- 보조배터리: mAh, Wh
- 액체류: ml, L, oz
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedSpecs:
    """OCR에서 추출된 수치 정보"""
    capacity_mah: Optional[int] = None
    capacity_wh: Optional[float] = None
    volume_ml: Optional[float] = None
    voltage_v: Optional[float] = None
    raw_text: str = ""

    @property
    def computed_wh(self) -> Optional[float]:
        """mAh + V가 있으면 Wh를 계산 (Wh = mAh × V / 1000)"""
        if self.capacity_wh:
            return self.capacity_wh
        if self.capacity_mah and self.voltage_v:
            return round(self.capacity_mah * self.voltage_v / 1000, 1)
        return None

    def to_dict(self) -> dict:
        return {
            "capacity_mah": self.capacity_mah,
            "capacity_wh": self.capacity_wh,
            "computed_wh": self.computed_wh,
            "volume_ml": self.volume_ml,
            "voltage_v": self.voltage_v,
        }


class SpecParser:
    """OCR 원문 텍스트에서 Regex로 수치를 파싱합니다."""

    # 패턴 우선순위: 구체적인 것부터 (mAh > Wh > ml > V)
    PATTERNS = {
        "capacity_mah": [
            r'(\d[\d,. ]*)\s*m[Aa][Hh]',           # 10000mAh, 10,000 mAh
            r'(\d[\d,. ]*)\s*밀리암페어',             # 한글 표기
        ],
        "capacity_wh": [
            r'(\d[\d,. ]*)\s*[Ww][Hh]',             # 37Wh, 37.0 Wh
            r'(\d[\d,. ]*)\s*와트시',                 # 한글 표기
        ],
        "volume_ml": [
            r'(\d[\d,. ]*)\s*[Mm][Ll]',              # 100ml, 100 mL
            r'(\d[\d,. ]*)\s*밀리리터',               # 한글 표기
            r'(\d[\d,. ]*)\s*[Ff][Ll]\.?\s*[Oo][Zz]', # 3.4 fl oz → ml 변환 필요
        ],
        "voltage_v": [
            r'(\d[\d,. ]*)\s*[Vv](?![Aa])',          # 3.7V (뒤에 A가 안 오는 경우만)
        ],
    }

    # fl oz → ml 변환 계수
    FL_OZ_TO_ML = 29.5735

    def extract(self, ocr_text: str) -> ParsedSpecs:
        """OCR 텍스트에서 수치를 추출하여 ParsedSpecs 반환"""
        specs = ParsedSpecs(raw_text=ocr_text)

        # mAh 추출
        for pattern in self.PATTERNS["capacity_mah"]:
            match = re.search(pattern, ocr_text)
            if match:
                val = self._clean_number(match.group(1))
                if val and 100 <= val <= 100000:  # 합리적인 범위
                    specs.capacity_mah = int(val)
                    break

        # Wh 추출
        for pattern in self.PATTERNS["capacity_wh"]:
            match = re.search(pattern, ocr_text)
            if match:
                val = self._clean_number(match.group(1))
                if val and 1 <= val <= 500:
                    specs.capacity_wh = float(val)
                    break

        # ml 추출
        for pattern in self.PATTERNS["volume_ml"]:
            match = re.search(pattern, ocr_text)
            if match:
                val = self._clean_number(match.group(1))
                if val:
                    # fl oz인 경우 ml로 변환
                    if "oz" in pattern.lower():
                        val = round(val * self.FL_OZ_TO_ML, 1)
                    if 1 <= val <= 10000:
                        specs.volume_ml = float(val)
                        break

        # 전압 추출
        for pattern in self.PATTERNS["voltage_v"]:
            match = re.search(pattern, ocr_text)
            if match:
                val = self._clean_number(match.group(1))
                if val and 1 <= val <= 50:
                    specs.voltage_v = float(val)
                    break

        return specs

    @staticmethod
    def _clean_number(raw: str) -> Optional[float]:
        """'10,000' → 10000, '3.7' → 3.7 등 숫자 정제"""
        try:
            cleaned = raw.replace(",", "").replace(" ", "").strip()
            return float(cleaned)
        except (ValueError, AttributeError):
            return None


# ── 규정 판단 로직 ─────────────────────────────────────────────────
class RegulationChecker:
    """추출된 수치를 기반으로 기내 반입 가능 여부를 판단합니다."""

    # 항공 보안 규정 기준값
    BATTERY_WH_CARRY_ON = 100     # 100Wh 이하: 기내 반입 가능
    BATTERY_WH_APPROVAL = 160    # 100~160Wh: 항공사 승인 필요
    LIQUID_ML_LIMIT = 100         # 액체류 100ml 이하

    def check_battery(self, specs: ParsedSpecs) -> dict:
        """보조배터리 규정 체크"""
        wh = specs.computed_wh

        if wh is None:
            return {
                "verdict": "확인 필요",
                "message": "배터리 용량(Wh)을 확인할 수 없습니다. 라벨을 다시 촬영해주세요.",
                "confidence": "low",
            }

        if wh <= self.BATTERY_WH_CARRY_ON:
            return {
                "verdict": "반입 가능 ✅",
                "message": f"배터리 용량 {wh}Wh — 100Wh 이하로 기내 반입 가능합니다.",
                "confidence": "high",
            }
        elif wh <= self.BATTERY_WH_APPROVAL:
            return {
                "verdict": "항공사 승인 필요 ⚠️",
                "message": f"배터리 용량 {wh}Wh — 100~160Wh 범위로 항공사 사전 승인이 필요합니다.",
                "confidence": "high",
            }
        else:
            return {
                "verdict": "반입 불가 ❌",
                "message": f"배터리 용량 {wh}Wh — 160Wh 초과로 기내 및 위탁 수하물 모두 반입 불가합니다.",
                "confidence": "high",
            }

    def check_liquid(self, specs: ParsedSpecs) -> dict:
        """액체류 규정 체크"""
        ml = specs.volume_ml

        if ml is None:
            return {
                "verdict": "확인 필요",
                "message": "용량(ml)을 확인할 수 없습니다. 라벨을 다시 촬영해주세요.",
                "confidence": "low",
            }

        if ml <= self.LIQUID_ML_LIMIT:
            return {
                "verdict": "반입 가능 ✅",
                "message": f"용량 {ml}ml — 100ml 이하로 투명 지퍼백에 담아 기내 반입 가능합니다.",
                "confidence": "high",
            }
        else:
            return {
                "verdict": "기내 반입 불가 ❌",
                "message": f"용량 {ml}ml — 100ml 초과로 기내 반입 불가합니다. 위탁 수하물로 부쳐주세요.",
                "confidence": "high",
            }
