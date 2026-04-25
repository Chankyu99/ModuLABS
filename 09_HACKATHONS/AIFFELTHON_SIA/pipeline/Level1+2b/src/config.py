"""
SIA 갈등 모니터링 파이프라인 - 설정파일
──────────────────────────────────────
- 1. 프로젝트 경로 설정
- 2. 데이터 필터링: CAMEO 코드, 모니터링 대상 국가
- 3. 칼만 필터 파라미터
- 4. 갈등 지수 산출 로직: AvgTone 가중치, EventCode 가중치
- 5. 리스크 레벨 임계값 및 대응 가이드
- 6. 지오코딩 블랙리스트
- 7. LLM 설정
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# ──────────────────────────────────────────────
# 1. 프로젝트 경로 설정
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "daily"
OUTPUT_DIR = PROJECT_ROOT / "output"


def _resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


MAIN_PATH = _resolve_existing_path(
    PROJECT_ROOT / "gdelt_main_2026.parquet",
    PROJECT_ROOT / "data" / "gdelt_main_2026.parquet",
    PROJECT_ROOT / "gdelt_main_final.parquet",
    PROJECT_ROOT / "data" / "gdelt_main_final.parquet",
)
URL_PATH = _resolve_existing_path(
    PROJECT_ROOT / "gdelt_url_final.parquet",
    PROJECT_ROOT / "data" / "gdelt_url_final.parquet",
    PROJECT_ROOT / "gdelt_url_2026.parquet",
    PROJECT_ROOT / "data" / "gdelt_url_2026.parquet",
)

# ──────────────────────────────────────────────
# 2. 데이터 필터링
# ──────────────────────────────────────────────

# 물리적 충돌과 관련된 CAMEO 코드 (15, 17, 18, 19, 20) 27개
CONFIRMED_CODES = [
150, 152, 154, 1712, 180, 181, 183, 1831, 1832,
1833, 1834, 186, 190, 191, 192, 193, 194,
195, 1951, 1952, 200, 201, 202, 203, 204, 2041, 2042
]

# 주요 모니터링 국가 (핵심 3국 + 대리전 지역 및 주요 이해관계국)
MONITORED_COUNTRIES = [
    'IR', 'IS',                            # 핵심 국가 (이란, 이스라엘)
    'IZ', 'LE', 'SY', 'YM',                # 분쟁 및 대리전 지역 (이라크, 레바논, 시리아, 예멘)
    'AE', 'SA', 'QA', 'KU', 'BA', 'MU'     # 주요 인접국 (UAE, 사우디, 카타르, 쿠웨이트, 바레인, 오만)
]

# ──────────────────────────────────────────────
# 3. 갈등 지수(Z_t) 산출 로직 (최종 업데이트: 0417)
# ──────────────────────────────────────────────

# 로지스틱 회귀 가중치 (1) (StandardScaler 적용 기준 - Mentions 사용)
# COEFS = {
#     'w_log_mentions': 0.0734,
#     'w_goldstein': -0.126,
#     'w_avgtone': -0.9528,
#     'w_avgtone_sq': -1.3528
# }

# # 로지스틱 회귀 가중치 (2) (StandardScaler 적용 기준 - Sources 사용)
# COEFS = {
#     'w_log_sources': 0.0970,    # 정보 출처의 다양성 가중치 (신뢰도)
#     'w_goldstein': -0.1226,     # 사건의 물리적 강도 가중치
#     'w_avgtone': -0.9307,       # 뉴스 논조의 기본 위험도
#     'w_avgtone_sq': -1.3231     # 극단적 논조에 대한 가중치
# }

# 로지스틱 회귀 가중치 (3) Log_Sources 및 Log_Mentions가 포함된 가중치
COEFS = {
    'w_log_sources': 0.1285,    # 정보 출처의 다양성 (가장 유의미)
    'w_log_mentions': -0.0396,  # 언급량 (출처보다 낮은 중요도, 음수 가중치로 조정)
    'w_avgtone': -0.9308,       # 뉴스 논조
    'w_avgtone_sq': -1.3229,    # 극단적 논조 가중
    'w_goldstein': -0.1221      # 사건 강도
}

# 분석에 필요한 변수 리스트
SCALING_FEATURES = ['Log_Mentions', 'Log_Sources', 'AvgTone', 'AvgTone_Sq', 'GoldsteinScale']

# 스케일러 저장 경로 (훈련 시 저장하고 예측 시 불러옴)
SCALER_PATH = PROJECT_ROOT / "models" / "standard_scaler.pkl"

# ──────────────────────────────────────────────
# 4. Kalman Filter 파라미터
# ──────────────────────────────────────────────

MIN_HISTORY = 30  # 칼만 필터 안정화를 위한 최소 관측 일수
KALMAN_MIN_VAR = 1.0    # 초기 최소 분산
KALMAN_Q = 0.001        # 프로세스 노이즈 (고정값, grid search 결과)
KALMAN_R = 10.0         # 관측 노이즈 (고정값, grid search 결과)
KALMAN_Q_RATIO = KALMAN_Q   # auto mode용 (manual_q=-1일 때 init_var에 곱함)
KALMAN_R_RATIO = KALMAN_R   # auto mode용 (manual_r=-1일 때 init_var에 곱함)
KALMAN_P0_RATIO = 2.0   # 초기 불확실성 계수

# ──────────────────────────────────────────────
# 5. 리스크 레벨 임계값 및 대응 가이드
# ──────────────────────────────────────────────

# 리스크 판단을 위한 임계치와 라벨을 순서대로 정의 (Numpy select용)
# 순서 짝 맞춰서 적기
RISK_THRESHOLDS = [3, 2, 1]    # 높은 것부터 쓰기
RISK_LEVELS_LIST = [3, 2, 1]
RISK_LABELS_LIST = ['위기', '위험', '주의']
RISK_GUIDES_LIST = [
    '즉시 대응',
    '정밀 분석',
    '모니터링'
]
RISK_EMOJIS_LIST = ['🛑', '🟠', '🟡']


# ──────────────────────────────────────────────
# 6. 지오코딩 블랙리스트 (조직명/무기명/지명 오류) -- 테스트 과정에서 잡히는 단어들 실시간 추가
# ──────────────────────────────────────────────
CITY_BLACKLIST = {
    'Basij',          # 이란 혁명수비대 민병대 (조직명)
    'Shahed',         # 이란 자폭 드론 이름 (무기명)
    'Hezbollah',      # 레바논 무장단체 (조직명)
    'Hamas',          # 팔레스타인 무장단체 (조직명)
    'Kurdistan',      # 지역명이 도시로 잡힘
    'Arabian Peninsula', # 반도 전체가 도시로 잡힘
    'As Iran',          # GDELT 파싱 오류
    'Sepah', # 이란 혁명수비대(IRGC)를 지칭하는 페르시아어 단어
    'Palestinian Red Crescent', # 팔레스타인 적십자사
    'Khaleej Times', # UAE 신문사 이름
    'Ministry of Foreig', 'Ministry Of Foreign Affairs',  # 외교부
    'Gaza', 'West Bank', 'Gaza City',   # 이란 전쟁과 직접적인 관련 없는 지역
    'Gulf News', 'Al Jazeera', 'Reuters', 'AP', 'AFP'  # 뉴스 매체명
}

# ──────────────────────────────────────────────
# 7. LLM 설정
# ──────────────────────────────────────────────
LLM_MODELS = [
    "gemini-2.5-flash-lite",
    'gemini-2.5-flash'    
]
LLM_TOP_N = 20          # llm 검증 대상 상위 도시 수 (run_experiment.py --top-k 기본값과 일치)
LLM_TOP_K_URLS = 5      # 도시당 초기 검증 기사 수

# llm 프롬프트 템플릿 (현재 로직에서는 사용 안하지만, 향후 활용 가능)
CAMEO_DEFINITION = {
    150: "Military/police power demonstration",
    152: "Increase military alert status",
    154: "Mobilize armed forces",
    1712: "Destroy property",
    180: "Unconventional violence",
    181: "Abduct, hijack, or take hostage",
    183: "Non-military bombing",
    1831: "Suicide bombing",
    1832: "Vehicular bombing",
    1833: "Roadside bombing",
    1834: "Location bombing",
    186: "Assassinate",
    190: "Conventional military force",
    191: "Blockade or restrict movement",
    192: "Occupy territory",
    193: "Fight with small arms and light weapons",
    194: "Fight with artillery and tanks",
    195: "Aerial weapons (General)",
    1951: "Precision-guided missiles",
    1952: "Drones/Remotely piloted weapons",
    200: "Unconventional mass violence",
    201: "Mass expulsion",
    202: "Mass killings",
    203: "Ethnic cleansing",
    204: "WMD use (General)",
    2041: "Chemical/Biological/Radiological weapons",
    2042: "Nuclear weapons"
}
