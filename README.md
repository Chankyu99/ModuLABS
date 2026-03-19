# ✈️ 기내뭐돼 (What Can I Bring?)

> 모두의연구소 데이터사이언티스트 7기 랭체인톤  
> **멀티모달 RAG 기반 국가/항공사별 수하물 반입 규정 AI 챗봇**

---

## 서비스 소개

**기내뭐돼**는 공식 법령 및 항공 보안 규정 원문 데이터를 바탕으로, 사용자가 원하는 물품의 기내/위탁 반입 가능 여부를 대화형으로 안내하는 RAG 챗봇입니다. 텍스트 질문뿐만 아니라 **제품 라벨 사진을 인식하여 자동으로 규정을 분석**하는 멀티모달 기능을 제공합니다.

- 🟢 **반입 가능** / 🟡 **조건부 가능** / 🔴 **반입 불가** 로 직관적인 판정 제공
- 📸 **멀티모달 인사이트**: YOLOv8 탐지 + OCR 기반 용량/모델명 자동 추출
- ⚠️ **신뢰도 시스템**: 보조배터리 등 리콜 대상 모델DB 대조 및 저신뢰도 구간 수동 입력 유도(Fallback)
- ⚡️ **초고속 응답 속도**: 추론 최적화(`reasoning_effort="low"`) 적용
- 🔀 **동적 대화 지원**: 대화 도중 실시간 노선/물품 정보 갱신(Dynamic Override)

---

## 프로젝트 구조

```
LANGCHAINTON_DS7/
├── app.py              # Streamlit 챗봇 UI (메인 실행 파일)
├── requirements.txt    # 의존성 패키지 목록
├── best.pt             # YOLOv8 학습 완료 가중치 모델
│
├── vision/             # 비전 파이프라인 (YOLO + OCR)
│   ├── ocr_pipeline.py # YOLO 탐지 및 EasyOCR 통합 제어
│   ├── spec_parser.py  # Regex V3 기반 수치 추출 및 규정 판단 로직
│   └── risk_model_db.py# 리콜 대상 노트북/태블릿 모델 데이터베이스
│
├── core/               # RAG 코어 로직
│   ├── bot_logic.py    # 슬롯필링, 검색, 생성 파이프라인
│   ├── ingest.py       # 데이터 임베딩 및 Vector DB 관리
│   └── ds7_rag/        # RAG 엔진 관련 모듈
│
├── data/               # 데이터셋 및 DB
│   ├── chroma_db/      # Vector Store 저장소
│   └── index_docstore_export.jsonl # 항공 규정 원문 데이터
│
├── scripts/            # 개발/평가용 유틸리티
│   ├── train_yolo.py   # YOLO 학습 스크립트
│   ├── evaluate_yolo.py# 모델 성능 평가 (mAP, FP/FN 분석)
│   └── ocr_benchmark.py# OCR 인식률 벤치마크 테스트
│
└── docs/               # 발표 자료 및 기술 문서
```

---

## 빠른 시작

### 1. 가상환경 세팅 (Python 3.9.6 권장)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 환경변수 설정

---

## 🔥 **주요 UX 향상 및 최적화 포인트**
- **레이턴시 극대화 (`reasoning_effort="low"`)**: 불필요한 LLM 내부 추론 루프를 강제로 짧게 끊어내어 답변 품질 저하 없이 전체 프로세스 속도를 약 2.5배 쾌속화 설계
- **초정밀 누락 슬롯 유도 (Targeted Asking)**: 단순히 "어디서 어디로 가시나요?" 대신 "🛬 도착하시는 국가를 알려주세요" 등 부족한 슬롯만을 핀포인트로 질문
- **LLM 환각(Hallucination) 방어벽**: 출발지를 유추하지 않고 무조건 `null` 로 엄격히 통제 + 미지원 노선(일본 등) 입력 시 **⚠️ 타국가 규정 경고 선출력** 방어 로직 탑재

---

## 기술 스택

- **LLM**:
  - ⚙️ **기본 파이프라인**: `gpt-5-mini` (`reasoning_effort` 튜닝을 통한 속도 최적화)
  - 🧠 **일반 지식 Fallback**: `gpt-5.2` (DB 미등재 물품에 대해 최상위 플래그십 AI로 깊이 있는 추론 제공)
- **Embedding**: `text-embedding-3-small` (OpenAI)
- **Vector DB**: ChromaDB
- **Framework**: LangChain
- **UI**: Streamlit
