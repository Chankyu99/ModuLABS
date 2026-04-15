"""
도시명 정규화 유틸리티
──────────────────────
- GDELT / GT / 수작업 문서 사이의 흔한 표기 차이를 흡수한다.
- 실제 다른 도시를 과도하게 합치지 않도록 "철자/동의어" 위주만 다룬다.
"""
from __future__ import annotations

import re


_CITY_ALIAS_MAP = {
    "teheran": "Tehran",
    "tehran": "Tehran",
    "isfahan": "Isfahan",
    "esfahan": "Isfahan",
    "beer sheba": "Beersheba",
    "beer sheva": "Beersheba",
    "be'er sheva": "Beersheba",
    "tel aviv yafo": "Tel Aviv",
    "tel aviv-yafo": "Tel Aviv",
    "kharg": "Kharg Island",
    "jurf al nasr": "Jurf al Sakhr",
    "jurf al-nasr": "Jurf al Sakhr",
    "jurf al sakhr": "Jurf al Sakhr",
    "kuwait": "Kuwait City",
}


def _normalize_key(value: str) -> str:
    key = str(value).strip().lower()
    key = key.split(",")[0].strip()
    key = re.sub(r"[\u2018\u2019\u201c\u201d']", "", key)
    key = re.sub(r"[-_/]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key


def normalize_city_name(value: str) -> str:
    """도시명을 canonical label로 정규화한다."""
    raw = str(value).strip()
    if not raw:
        return raw

    key = _normalize_key(raw)
    alias = _CITY_ALIAS_MAP.get(key)
    if alias is not None:
        return alias

    # alias가 없으면 원본 표기를 최대한 보존하되, 콤마 이후 세부 표기만 제거한다.
    return raw.split(",")[0].strip()


def normalize_city_key(value: str) -> str:
    """비교/조인용 lowercase key."""
    return normalize_city_name(value).strip().lower()
