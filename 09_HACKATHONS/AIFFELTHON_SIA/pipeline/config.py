"""
SIA 갈등 모니터링 파이프라인 - 설정파일
──────────────────────────────────────
- CAMEO 분쟁 코드 (폭력/군사 행동)
- 모니터링 대상 국가 및 전역(Theater of War)
- 점수 가산치 및 칼만 필터 파라미터
"""

import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────
# 1. 프로젝트 경로 설정
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "daily"
OUTPUT_DIR = PROJECT_ROOT / "output"
PARQUET_PATH = PROJECT_ROOT / "gdelt_main_final.parquet"

# ──────────────────────────────────────────────
# 2. 데이터 필터링 조건
# ──────────────────────────────────────────────

# 물리적 충돌과 관련된 CAMEO 코드 (Root 15, 17, 18, 19, 20)
CONFIRMED_CODES = [
    '150', '152', '1712', '180', '181', '183', '1831', '1832', 
    '1833', '1834', '186', '190', '191', '192', '193', '194', 
    '195', '1951', '1952', '200', '201', '202', '204', '2042'
]

# 주요 모니터링 국가 (핵심 3국 + 대리전 지역 및 주요 이해관계국)
MONITORED_COUNTRIES = [
    'IRN', 'USA', 'ISR',               # 핵심 국가
    'IRQ', 'LBN', 'PSE', 'SYR', 'YEM',  # 분쟁 및 대리전 지역
    'ARE', 'SAU', 'QAT', 'KWT'          # 주요 인접국
]

# ──────────────────────────────────────────────
# 3. 갈등 지수(I) 산출 로직
# ──────────────────────────────────────────────

def tone_weight(avg_tone: float) -> float:
    """기사 어조(AvgTone)에 따른 심각도 가중치 반환"""
    if avg_tone < -15: return 0.5  # 극단적 부정 (자극적 보도 가능성)
    if avg_tone < -5:  return 1.0  # 위험 구간 (실제 갈등 징후가 가장 뚜렷함)
    if avg_tone < 0:   return 0.3  # 약한 부정
    return 0.1                     # 중립 또는 긍정 (단순 배경 소음)

# ──────────────────────────────────────────────
# 4. 칼만 필터(Kalman Filter) 파라미터
# ──────────────────────────────────────────────
KALMAN_Q_RATIO = 0.01   # 프로세스 노이즈 (변화에 대한 민감도)
KALMAN_R_RATIO = 1.0    # 관측 노이즈 (데이터에 대한 신뢰도)
KALMAN_P0_RATIO = 2.0   # 초기 불확실성 계수

# ──────────────────────────────────────────────
# 5. 리스크 레벨 및 대응 가이드
# ──────────────────────────────────────────────
MIN_HISTORY = 30  # 칼만 필터 안정화를 위한 최소 관측 일수

RISK_LEVELS = {
    3: {
        'label': '위기 (RED)',
        'threshold': 150.0,
        'emoji': '🛑',
        'guide': '대규모 충돌/공격 포착. 즉시 위성 촬영 스케줄링 필수.'
    },
    2: {
        'label': '위험 (ORANGE)',
        'threshold': 20.0,
        'emoji': '🟠',
        'guide': '물리적 교전 확인. 우선순위 위성 촬영 및 정밀 분석 착수.'
    },
    1: {
        'label': '주의 (YELLOW)',
        'threshold': 5.0,
        'emoji': '🟡',
        'guide': '긴장 고조 및 국지적 징후 탐지. ROI 모니터링 명단 추가.'
    },
    0: {
        'label': '정상 (BLUE)',
        'threshold': -np.inf,
        'emoji': '🔵',
        'guide': '평시 수준의 뉴스 흐름 유지.'
    }
}

def get_risk_level(z_score: float) -> dict:
    """Z-Score에 따른 리스크 등급 정보 반환"""
    for level in sorted(RISK_LEVELS.keys(), reverse=True):
        if z_score >= RISK_LEVELS[level]['threshold']:
            res = RISK_LEVELS[level].copy()
            res['level'] = level
            return res
    return RISK_LEVELS[0]
