"""
risk_model_db.py
================
항공 기내 반입이 금지되거나 제한된 특정 기기 모델 데이터베이스.
노트북/태블릿 탐지 시 해당 기기가 위험 모델인지 조회합니다.
"""


# ── 반입 금지/제한 기기 목록 ──────────────────────────────────────
BANNED_MODELS = [
    # ─ 노트북
    {
        "category": "laptop",
        "brand": "Apple",
        "model_keywords": ["macbook pro", "15-inch", "15인치"],
        "year_range": (2015, 2017),
        "reason": "배터리 과열 리콜 대상 (2019년 Apple 자발적 리콜). 일부 항공사 기내 반입 제한.",
        "severity": "warning",  # warning: 주의, banned: 금지
        "source": "미국 FAA / Apple 리콜 프로그램",
    },
    {
        "category": "laptop",
        "brand": "Apple",
        "model_keywords": ["macbook pro", "15-inch", "15인치"],
        "year_range": (2015, 2017),
        "reason": "배터리 결함으로 인한 화재 위험. EASA(유럽항공안전청) 경고 발령.",
        "severity": "banned",
        "source": "EASA Safety Information Bulletin 2019-10",
    },
    # ─ 태블릿/스마트폰
    {
        "category": "tablet",
        "brand": "Samsung",
        "model_keywords": ["galaxy note 7", "갤럭시 노트 7", "note7", "sm-n930"],
        "year_range": (2016, 2016),
        "reason": "배터리 발화 위험으로 전 세계 항공기 기내·위탁 수하물 탑재 전면 금지.",
        "severity": "banned",
        "source": "미국 DOT / FAA 긴급 명령 (2016.10)",
    },
    {
        "category": "laptop",
        "brand": "HP",
        "model_keywords": ["probook 640", "probook 645", "probook 650", "probook 655"],
        "year_range": (2017, 2018),
        "reason": "HP 배터리 리콜 프로그램 대상. 항공사별 제한 가능.",
        "severity": "warning",
        "source": "미국 CPSC 리콜 공지",
    },
    {
        "category": "laptop",
        "brand": "Lenovo",
        "model_keywords": ["thinkpad x1 carbon", "5th gen"],
        "year_range": (2017, 2017),
        "reason": "배터리 과열 리콜 대상. 항공사별 제한 가능.",
        "severity": "warning",
        "source": "Lenovo 자발적 리콜",
    },
]


class RiskModelDB:
    """탐지된 기기가 위험 모델인지 조회합니다."""

    def __init__(self):
        self.db = BANNED_MODELS

    def check(self, class_name: str, ocr_text: str = "") -> dict:
        """
        탐지된 기기의 클래스명과 OCR 텍스트를 기반으로 위험 모델 여부를 조회합니다.

        Args:
            class_name: YOLO가 탐지한 클래스명 ('laptop', 'tablet')
            ocr_text: 해당 영역에서 읽은 OCR 텍스트 (모델명 식별용)

        Returns:
            dict: verdict, message, matched_models 등
        """
        ocr_lower = ocr_text.lower()
        matched = []

        for entry in self.db:
            if entry["category"] != class_name:
                continue

            # 키워드 매칭: 모델 키워드 중 하나라도 OCR 텍스트에 포함되면 매치
            keyword_hit = any(kw.lower() in ocr_lower for kw in entry["model_keywords"])
            if keyword_hit:
                matched.append(entry)

        if matched:
            # 가장 심각한 것 우선
            worst = max(matched, key=lambda x: 1 if x["severity"] == "banned" else 0)
            severity_icon = "❌" if worst["severity"] == "banned" else "⚠️"
            return {
                "verdict": f"위험 기기 감지 {severity_icon}",
                "message": f"{worst['brand']} {', '.join(worst['model_keywords'][:2])} — {worst['reason']}",
                "severity": worst["severity"],
                "matched_models": [
                    {"brand": m["brand"], "keywords": m["model_keywords"],
                     "reason": m["reason"], "source": m["source"]}
                    for m in matched
                ],
                "confidence": "high",
            }
        else:
            return {
                "verdict": f"{class_name} 탐지 — 특별 제한 없음 ✅",
                "message": f"{class_name}이(가) 감지되었습니다. 현재 알려진 반입 금지 모델에 해당하지 않습니다.",
                "severity": "ok",
                "matched_models": [],
                "confidence": "medium",
                "note": "기기 모델명을 정확히 식별하지 못했을 수 있습니다. 불안하시다면 항공사에 직접 문의하세요.",
            }

    def get_all_banned(self) -> list:
        """전체 금지/제한 모델 목록 반환 (UI 표시용)"""
        return [
            {
                "category": m["category"],
                "brand": m["brand"],
                "description": ", ".join(m["model_keywords"][:3]),
                "severity": m["severity"],
                "reason": m["reason"],
            }
            for m in self.db
        ]
