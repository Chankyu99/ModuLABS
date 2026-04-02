"""
파이프라인 설정값
- 확정 CAMEO 코드 21개
- Triad 국가
- AvgTone 가중치 함수
- 칼만 필터 / Z-score 파라미터
"""

from pathlib import Path

# ──────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "daily"
OUTPUT_DIR = PROJECT_ROOT / "output"
PARQUET_PATH = PROJECT_ROOT / "gdelt_main_final.parquet"

# ──────────────────────────────────────────────
# GDELT 필터 조건
# ──────────────────────────────────────────────
CONFIRMED_CODES = [
    # Root 15 — 군사태세 (선제 징후)
    '150', '152',
    # Root 17 — 강압
    '1712',
    # Root 18 — 비정규 폭력
    '181', '183', '1831', '1832', '1833', '1834', '186',
    # Root 19 — 정규 교전
    '191', '192', '193', '194', '195', '1951', '1952',
    # Root 20 — 대량 폭력
    '201', '202', '204', '2042',
]

TRIAD_COUNTRIES = ['IRN', 'USA', 'ISR']

# ──────────────────────────────────────────────
# 갈등지수 I = Σ(NumMentions × W(AvgTone))
# ──────────────────────────────────────────────
# (하한, 상한, 가중치)
TONE_WEIGHTS = [
    (-100, -15, 0.5),   # 극단 부정 — 자극적 보도 가능성
    (-15,   -5, 1.0),   # Critical Zone — 가장 신뢰
    ( -5,    0, 0.3),   # 경미
    (  0,  100, 0.1),   # 중립/긍정 — 배경 소음
]

def tone_weight(avg_tone: float) -> float:
    """AvgTone 값에 대응하는 가중치를 반환"""
    for lo, hi, w in TONE_WEIGHTS:
        if lo <= avg_tone < hi:
            return w
    return 0.1  # fallback


# ──────────────────────────────────────────────
# 칼만 필터 파라미터
# ──────────────────────────────────────────────
KALMAN_Q_RATIO = 0.01   # Q = 초기 30일 분산 × 이 비율
KALMAN_R_RATIO = 1.0    # R = 초기 30일 분산 × 이 비율
KALMAN_P0_RATIO = 2.0   # 초기 불확실성 = R × 이 비율

# ──────────────────────────────────────────────
# 이상탐지 파라미터
# ──────────────────────────────────────────────
ROLLING_WINDOW = 30     # Rolling Z-score 윈도우 (일)
Z_THRESHOLD = 2.0       # 이상 판정 임계치
MIN_HISTORY = 30        # 최소 과거 데이터 (일) — 이 이하면 판정 보류
