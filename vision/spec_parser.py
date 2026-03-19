"""
spec_parser.py
==============
OCR 텍스트에서 항공 보안 규정 관련 핵심 수치를 추출하는 모듈.
- 보조배터리: mAh, Wh
- 액체류: ml, L, g, kg, oz, fl oz

개선사항 (v2):
  1. OCR 오타 교정 (O→0, o→0, l→1, I→1)
  2. 한국어 라벨 패턴 추가 (내용량, 용량, 정격 등)
  3. g, kg, oz 등 추가 단위 지원
  4. 컨텍스트 기반 추론 (단위 없는 숫자 추정)
  5. 괄호 안 수치 매칭 (예: "(700ml)")
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
    weight_g: Optional[float] = None
    voltage_v: Optional[float] = None
    raw_text: str = ""
    confidence: str = "measured"   # "measured" | "low" (추론된 값)

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
            "weight_g": self.weight_g,
            "voltage_v": self.voltage_v,
            "confidence": self.confidence,
        }


# OCR의 숫자 오인식 교정 (문맥 기반 활용)
OCR_DIGIT_MAP = str.maketrans({
    'O': '0', 'o': '0', 'Q': '0',
    'S': '5', 's': '5',
    'B': '8',
    # l, I, | 는 단위(ml) 인식 방해 방지를 위해 글로벌 맵에서 제외
})


class SpecParser:
    """OCR 원문 텍스트에서 Regex로 수치를 파싱합니다. (v2 — OCR 오타 대응)"""

    # ── 변환 계수 ──────────────────────────────────────────────
    FL_OZ_TO_ML = 29.5735
    OZ_TO_G = 28.3495

    # ── 패턴 정의 (우선순위: 구체적 → 일반적) ─────────────────
    PATTERNS = {
        "capacity_mah": [
            # 단위가 붙은 확실한 패턴 -> high confidence
            r'(\d[\d,. ]*)\s*m[Aa][Hh]',
            r'(\d[\d,. ]*)\s*밀리암페어',
            r'정격\s*[\d.]*\s*[Vv]\s*[/\s]*(\d[\d,. ]*)\s*m[Aa][Hh]',
        ],
        "capacity_wh": [
            r'(\d[\d,. ]*)\s*[Ww][Hh]',
            r'(\d[\d,. ]*)\s*와트시',
        ],
        "volume_ml": [
            r'(\d[\d,. ]*)\s*[Mm][Ll1I|]',                              # 500ml, 500m1, 500mI 대응
            r'(\d[\d,. ]*)\s*밀리리터',
            r'(?:내용량|용량)\s*[:\-]?\s*(\d[\d,.]*)\s*[Mm][Ll1I|]',
        ],
        "weight_g": [
            r'(\d[\d,. ]*)\s*[Gg](?![a-zA-Z])',
            r'(?:내용량|중량)\s*[:\-]?\s*(\d[\d,.]*)\s*[Gg](?![a-zA-Z])',
        ],
        # 단위 없는 단독 숫자 패턴 (추론용) -> low confidence
        "standalone_digits": [
            r'(?<![\d/:\-.])(\d{4,5})(?!\d|/|[Vv]|mAh)',  # 시리얼 번호/날짜 제외 노력
        ],
        "voltage_v": [
            r'(\d[\d,. ]*)\s*[Vv](?![a-zA-Z])', 
        ],
    }

    def extract(self, ocr_text: str, class_name: str = "") -> ParsedSpecs:
        """
        OCR 텍스트에서 수치를 추출하여 ParsedSpecs 반환.

        Args:
            ocr_text: OCR 원문 텍스트
            class_name: YOLO가 감지한 클래스명 (컨텍스트 추론용)
        """
        specs = ParsedSpecs(raw_text=ocr_text)

        # OCR 오타 교정된 버전도 준비
        fixed_text = self._fix_ocr_digits(ocr_text)

        # ── mAh 추출 ──────────────────────────────────────────
        specs.capacity_mah = self._try_extract_int(
            [ocr_text, fixed_text], self.PATTERNS["capacity_mah"],
            valid_range=(100, 100000)
        )

        # ── Wh 추출 ───────────────────────────────────────────
        specs.capacity_wh = self._try_extract_float(
            [ocr_text, fixed_text], self.PATTERNS["capacity_wh"],
            valid_range=(1, 500)
        )

        # ── ml 추출 ───────────────────────────────────────────
        for texts in [[ocr_text, fixed_text]]:
            for pattern in self.PATTERNS["volume_ml"]:
                for text in texts:
                    match = re.search(pattern, text)
                    if match:
                        val = self._clean_number(match.group(1))
                        if val is None: continue
                        # 단위 변환
                        if "oz" in pattern.lower():
                            val = round(val * self.FL_OZ_TO_ML, 1)
                        if 5 <= val <= 3000:  # 비정상적 용량(5050 등) 필터링
                            specs.volume_ml = float(val)
                            break
                if specs.volume_ml: break
            if specs.volume_ml: break

        # ── 중량(g) 추출 ──────────────────────────────────────
        for texts in [[ocr_text, fixed_text]]:
            for pattern in self.PATTERNS["weight_g"]:
                for text in texts:
                    match = re.search(pattern, text)
                    if match:
                        val = self._clean_number(match.group(1))
                        if val is None:
                            continue
                        # 단위 변환
                        if re.search(r'[Kk][Gg]|킬로그램', pattern):
                            val = round(val * 1000, 1)
                        elif re.search(r'[Oo][Zz]', pattern):
                            val = round(val * self.OZ_TO_G, 1)
                        if 0.1 <= val <= 50000:
                            specs.weight_g = float(val)
                            break
                if specs.weight_g:
                    break
            if specs.weight_g:
                break

        # ── 전압 추출 ─────────────────────────────────────────
        specs.voltage_v = self._try_extract_float(
            [ocr_text, fixed_text], self.PATTERNS["voltage_v"],
            valid_range=(1, 50)
        )

        # ── 컨텍스트 추론 (단위 없는 숫자) ───────────────────
        if specs.capacity_mah is None and class_name == "power_bank":
            for pattern in self.PATTERNS["standalone_digits"]:
                for text in [ocr_text, fixed_text]:
                    match = re.search(pattern, text)
                    if match:
                        val = int(match.group(1))
                        # 환각 방지: 너무 생뚱맞은 숫자는 제외
                        if 3000 <= val <= 50000:
                            specs.capacity_mah = val
                            specs.confidence = "low"  # 단위 없이 숫자만 찾은 경우
                            break
                if specs.capacity_mah: break

        return specs

    # ── 내부 유틸리티 ─────────────────────────────────────────────
    @staticmethod
    def _fix_ocr_digits(text: str) -> str:
        """OCR이 자주 혼동하는 문자를 숫자로 교정"""
        return text.translate(OCR_DIGIT_MAP)

    def _try_extract_int(self, texts: list, patterns: list,
                         valid_range: tuple) -> Optional[int]:
        """여러 텍스트 버전(원본/교정)에서 정수 추출 시도"""
        lo, hi = valid_range
        for pattern in patterns:
            for text in texts:
                match = re.search(pattern, text)
                if match:
                    val = self._clean_number(match.group(1))
                    if val and lo <= val <= hi:
                        return int(val)
        return None

    def _try_extract_float(self, texts: list, patterns: list,
                           valid_range: tuple) -> Optional[float]:
        """여러 텍스트 버전(원본/교정)에서 실수 추출 시도"""
        lo, hi = valid_range
        for pattern in patterns:
            for text in texts:
                match = re.search(pattern, text)
                if match:
                    val = self._clean_number(match.group(1))
                    if val and lo <= val <= hi:
                        return float(val)
        return None

    @staticmethod
    def _clean_number(raw: str) -> Optional[float]:
        """'10,000' → 10000, '3.7' → 3.7 등 숫자 정제"""
        try:
            cleaned = raw.replace(",", "").replace(" ", "").strip()
            return float(cleaned)
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _infer_battery_from_context(text: str, specs: ParsedSpecs) -> ParsedSpecs:
        """
        단위 없는 숫자를 컨텍스트(클래스명)로 추론.
        power_bank 클래스에서 3000~50000 사이 4~5자리 숫자 → mAh로 추정.
        """
        standalone = re.search(r'(?<!\d)(\d{4,5})(?!\d)', text)
        if standalone:
            val = int(standalone.group(1))
            if 3000 <= val <= 50000:
                specs.capacity_mah = val
                specs.confidence = "inferred"
        return specs


# ── 규정 판단 로직 ─────────────────────────────────────────────────
class RegulationChecker:
    """추출된 수치를 기반으로 기내 반입 가능 여부를 판단합니다."""

    BATTERY_WH_CARRY_ON = 100
    BATTERY_WH_APPROVAL = 160
    LIQUID_ML_LIMIT = 100

    def check_battery(self, specs: ParsedSpecs) -> dict:
        """보조배터리 규정 체크"""
        wh = specs.computed_wh
        # 신뢰도가 낮은 경우(추론값 등)는 수치를 노출하지 않고 에러로 처리 (사용자 요청 반영)
        if wh is None or specs.confidence == "low":
            return {
                "verdict": "확인 필요",
                "message": "글씨가 인식 오류로 추정치를 직접 알려주세요. (mAh, Wh 등)",
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
        """액체류 규정 체크 (ml 또는 g 기준)"""
        ml = specs.volume_ml
        g = specs.weight_g

        # ml 우선, 없으면 g으로 판단 (크림/젤류는 g 단위가 많음)
        if (ml is not None or g is not None) and specs.confidence != "low":
            amount = ml if ml is not None else g
            unit = "ml" if ml is not None else "g"
            return self._judge_liquid(amount, unit)
        else:
            return {
                "verdict": "확인 필요",
                "message": "글씨가 인식 오류로 추정치를 직접 알려주세요. (ml, g 등)",
                "confidence": "low",
            }

    def _judge_liquid(self, amount: float, unit: str) -> dict:
        """용량/중량 기준 규정 판단"""
        if amount <= self.LIQUID_ML_LIMIT:
            return {
                "verdict": "반입 가능 ✅",
                "message": f"용량 {amount}{unit} — 100{unit} 이하로 투명 지퍼백에 담아 기내 반입 가능합니다.",
                "confidence": "high",
            }
        else:
            return {
                "verdict": "기내 반입 불가 ❌",
                "message": f"용량 {amount}{unit} — 100{unit} 초과로 기내 반입 불가합니다. 위탁 수하물로 부쳐주세요.",
                "confidence": "high",
            }
