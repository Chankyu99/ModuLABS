"""
SIA 갈등 모니터링 파이프라인 - 설정파일
──────────────────────────────────────
- CAMEO 분쟁 코드 (폭력/군사 행동)
- 모니터링 대상 국가 및 전역(Theater of War)
- 점수 가산치 및 칼만 필터 파라미터
"""

import numpy as np
from pathlib import Path

# 1. 프로젝트 경로 설정

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "daily"
OUTPUT_DIR = PROJECT_ROOT / "output"
PARQUET_PATH = PROJECT_ROOT / "gdelt_main_final.parquet"
LLM_CACHE_DIR = PROJECT_ROOT / "data" / "llm_cache"
LLM_CACHE_TTL_DAYS = 7          # 캐시 자동 만료 기간 (일)
LLM_PROMPT_VERSION = "v1"       # 프롬프트 변경 시 수동으로 올려 캐시 무효화

# 2. 데이터 필터링 조건

# 물리적 충돌과 관련된 CAMEO 코드 (Root 15, 17, 18, 19, 20)
CONFIRMED_CODES = [
    150, 152, 1712, 180, 181, 183, 1831, 1832, 
    1833, 1834, 186, 190, 191, 192, 193, 194, 
    195, 1951, 1952, 200, 201, 202, 204, 2042
]

# 주요 모니터링 국가 (핵심 3국 + 대리전 지역 및 주요 이해관계국)
MONITORED_COUNTRIES = [
    'IRN', 'USA', 'ISR',               # 핵심 국가
    'IRQ', 'LBN', 'PSE', 'SYR', 'YEM',  # 분쟁 및 대리전 지역
    'ARE', 'SAU', 'QAT', 'KWT'          # 주요 인접국
]

# ActionGeo_* 컬럼은 Actor 국가 코드와 다른 위치 코드 체계를 쓴다.
# 전장/타격 가능 권역만 허용해 unrelated third-country city 유입을 줄인다.
ACTION_GEO_ALLOWED_COUNTRIES = [
    'IR',  # Iran
    'IS',  # Israel
    'IZ',  # Iraq
    'LE',  # Lebanon
    'SY',  # Syria
    'YM',  # Yemen
    'AE',  # UAE
    'SA',  # Saudi Arabia
    'QA',  # Qatar
    'KU',  # Kuwait
    'BA',  # Bahrain
    'MU',  # Oman
    'JO',  # Jordan
    'WE',  # West Bank
    'EG',  # Egypt / Sinai 접근권
    'GR',  # Akrotiri/Cyprus 계열이 이 코드로 들어오는 사례 대응
]

# 3. 갈등 지수(I) 산출 로직
# 로지스틱 회귀로 도출한 per-event 가중치
# 연속형 변수(log(NumSources), AvgTone, AvgTone², GoldsteinScale)는 StandardScaler로
# 표준화(Z-score)된 뒤 아래 계수와 곱해진다. EventRootCode는 GoldsteinScale과 1:1
# 매핑되므로 제외했고, NumMentions는 p-value가 0.05를 넘어 제외했다.
LOGIT_WEIGHTS = {
    "const":       -2.9137,
    "log_sources":  0.0970,
    "avg_tone":    -0.9307,
    "avg_tone_sq": -1.3231,
    "goldstein":   -0.1226,
}

# --- 이전 계단형 가중치 (로지스틱 회귀 도입 이전) ---
# def tone_weight(avg_tone: float) -> float:
#     """기사 어조(AvgTone)에 따른 심각도 가중치 반환"""
#     if avg_tone < -15: return 0.5
#     if avg_tone < -5:  return 1.0
#     if avg_tone < 0:   return 0.3
#     return 0.1
#
# def source_count_weight(num_sources: float) -> float:
#     """단일 매체 보도는 남기되 soft penalty를 준다."""
#     if num_sources <= 0:
#         return 0.0
#     if num_sources < 2:
#         return 0.60
#     if num_sources < 3:
#         return 0.85
#     return 1.0

# 4. 칼만 필터(Kalman Filter) 파라미터
# Spatial GT 기준으로는 평시보다 변화 폭을 더 빠르게 따라가도록 Q를 키우고,
# 관측을 조금 더 신뢰하는 쪽이 성능이 안정적이었다.
KALMAN_Q_RATIO = 0.10   # 프로세스 노이즈 (변화에 대한 민감도)
KALMAN_R_RATIO = 0.25   # 관측 노이즈 (데이터에 대한 신뢰도)
KALMAN_P0_RATIO = 2.0   # 초기 불확실성 계수
KALMAN_MIN_INIT_VAR = 1.0  # 초기 분산 하한선 (0 채우기 시 노이즈 증폭 방지)

# --- 이전 EventRootCode 기반 가산치 맵 (로지스틱 회귀 분석 결과 GoldsteinScale과
# --- 1:1 매핑으로 드러나 다중공선성 방지를 위해 제외) ---
# EVENT_WEIGHT_MAP = {
#     20: 1.0,
#     19: 1.0,
#     18: 0.9,
#     17: 0.8,
#     15: 0.7
# }

# 5. 리스크 레벨 및 대응 가이드
MIN_HISTORY = 30  # 칼만 필터 안정화를 위한 최소 관측 일수


LLM_TOP_N = 20          # 정밀 출력 후보(10~15개)를 충분히 커버하기 위한 상위 도시 수
LLM_TOP_K_URLS = 5      # 도시당 초기 검증 기사 수
LLM_PREFETCH_URLS = 8   # 후보당 먼저 가져와 볼 기사 수
LLM_MIN_EXACT_SUPPORT = 2  # actionable 판정을 위한 최소 exact-city 기사 수
LLM_ALLOW_SINGLE_ARTICLE_EXACT = True  # 기사 1건만 확보된 경우 exact-city면 통과 허용
LLM_ALLOW_STRATEGIC_SINGLE_SUPPORT = True  # 전략 표적은 exact/nearby 1건 + 명시적 표적 언급이면 통과 허용
LLM_ALLOW_ROI_PRIOR_SINGLE_SUPPORT = True  # ROI prior 도시는 exact/nearby 1건으로 통과 허용
LLM_STRATEGIC_KEYWORDS = [
    "airport",
    "air base",
    "airbase",
    "military base",
    "naval base",
    "missile",
    "nuclear",
    "reactor",
    "heavy water",
    "refinery",
    "gas facility",
    "terminal",
    "port",
    "shipyard",
    "bridge",
    "steel",
    "factory",
    "plant",
    "industrial",
    "radar",
    "air defense",
    "command center",
    "headquarters",
    "drone",
    "shipyard",
]

# Precision 우선 출력 정책
REPORT_MAX_ALERTS = 15
REPORT_MIN_CONFLICT_INDEX = 3.0
REPORT_MIN_INNOV_Z = 1.5

# 7. 지오코딩 블랙리스트 (조직명/무기명/지명 오류) -- 테스트 과정에서 잡히는 단어들 실시간 추가
CITY_BLACKLIST = {
    'Basij',          # 이란 혁명수비대 민병대 (조직명)
    'Shahed',         # 이란 자폭 드론 이름 (무기명)
    'Hezbollah',      # 레바논 무장단체 (조직명)
    'Hamas',          # 팔레스타인 무장단체 (조직명)
    'Kurdistan',      # 지역명이 도시로 잡힘
    'Arabian Peninsula', # 반도 전체가 도시로 잡힘
    'As Iran', # 이건 뭐임?
    'Sepah', # 이란 혁명수비대(IRGC)를 지칭하는 페르시아어 단어
    'Palestinian Red Crescent', # 팔레스타인 적십자사
    'Ministry Of Foreign Affairs' # 외교부
}


# 8. Level 2a: 위성 촬영 스케줄 설정
# 위성 선택 5대 기준: 센서 비율 / 해상도 / 군집 여부 / 데이터 접근성 / 활성 상태

SATELLITES = [
    {
        "name": "SpaceEye-T",  "norad_id": 63229,
        "type": "optical",     "swath_km": 12,    "resolution_m": 0.25,
        "off_nadir_deg": 45,   "orbit": "SSO",    "altitude_km": 510,
        "priority": 1,
    },
    {
        "name": "KOMPSAT-7",   "norad_id": 66820,
        "type": "optical",     "swath_km": 15,    "resolution_m": 0.30,
        "off_nadir_deg": 30,   "orbit": "SSO",    "altitude_km": 570,
        "priority": 2,
    },
    {
        "name": "SkySat-C12",  "norad_id": 43797,
        "type": "optical",     "swath_km": 5.9,   "resolution_m": 0.50,
        "off_nadir_deg": 30,   "orbit": "SSO",    "altitude_km": 500,
        "priority": 3,
    },
    {
        "name": "Sentinel-2A", "norad_id": 40697,
        "type": "optical",     "swath_km": 290,   "resolution_m": 10,
        "off_nadir_deg": None, "orbit": "SSO",    "altitude_km": 786,
        "priority": 4,
    },
    {
        "name": "ICEYE-X2",    "norad_id": 43800,
        "type": "sar",         "swath_km": 30,    "resolution_m": 1,
        "off_nadir_deg": 35,   "orbit": "SSO",    "altitude_km": 570,
        "priority": 5,
    },
]

# 기상 판별 설정
CLOUD_THRESHOLD = 50        # 구름량 50% 초과 시 EO 촬영 부적합
PREDICTION_HOURS = 168      # 7일 관측 창 확보를 위한 168시간 예측
MIN_ELEVATION_DEG = 20.0    # 최소 앙각 (기존 10도 -> 20도 상향, 대기 왜곡/그림자 방지)

# TLE 캐시
TLE_CACHE_DIR = PROJECT_ROOT / "data" / "tle"

# ROI 도시 좌표 (Level 1 탐지 대상 + 핵심 인프라)
ROI_CITIES = {
    "Isfahan":      {"lat": 32.6546, "lon": 51.6680},
    "Natanz":       {"lat": 33.5130, "lon": 51.9220},
    "Bushehr":      {"lat": 28.9684, "lon": 50.8385},
    "Tehran":       {"lat": 35.6892, "lon": 51.3890},
    "Tabriz":       {"lat": 38.0800, "lon": 46.2919},
    "Kharg Island": {"lat": 29.2333, "lon": 50.3167},
    "Dimona":       {"lat": 31.0700, "lon": 35.2100},
    "Beirut":       {"lat": 33.8938, "lon": 35.5018},
    "Baghdad":      {"lat": 33.3152, "lon": 44.3661},
    "Gaza":         {"lat": 31.5000, "lon": 34.4667},
    "Tel Aviv":     {"lat": 32.0853, "lon": 34.7818},
    "Minab":        {"lat": 27.1064, "lon": 57.0850},
    "Ras Laffan":   {"lat": 25.9300, "lon": 51.5300},
    "Fujairah":     {"lat": 25.1288, "lon": 56.3265},
    "Dubai":        {"lat": 25.2048, "lon": 55.2708},
}
