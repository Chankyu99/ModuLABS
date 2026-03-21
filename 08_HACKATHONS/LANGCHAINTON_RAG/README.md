# ✈️ 기내뭐돼?

> 모두의연구소 데이터사이언티스트 7기 LANGCHAINTHON  
> **RAG 기반 국가/항공사별 수하물 반입 규정 AI 챗봇**

---

## 목차

1. [프로젝트 배경](#-프로젝트-배경)
2. [서비스 소개](#-서비스-소개)
3. [기술적 도전과 해결](#-기술적-도전과-해결)
4. [RAG 파이프라인](#-rag-파이프라인)
5. [프로젝트 구조](#-프로젝트-구조)
6. [빠른 시작](#-빠른-시작)
7. [기술 스택](#-기술-스택)
8. [팀원 소개](#-팀원-소개)

---

## 📌 프로젝트 배경

2024년 기준 해외여행객 **2,872만 명** 중 인천공항에서 반입금지 물품이 적발된 건수는 약 **500만 건** — 전체 출발 승객 100명 중 약 **17.8명**이 반입금지 물품을 소지한 채 공항에 도착합니다.

기존 항공사 챗봇과 범용 AI의 한계는 명확했습니다.

| 기존 서비스 | 문제점 |
|---|---|
| 대한항공 챗봇 | 미국 규정을 모른 채 추측으로 답변, 응답 10초 이상 소요 |
| Google Gemini | 답변이 길고(22줄+), 출처 불명확, 2026년 신규 규정 미반영 |

**기내뭐돼**는 공식 법령 및 항공 보안 규정 원문 데이터를 기반으로, 정확하고 빠르게 반입 가능 여부를 안내하는 RAG 챗봇입니다.

---

## 🛫 서비스 소개

- 🟢 **반입 가능** / 🟡 **조건부 가능** / 🔴 **반입 불가** 로 직관적인 이모지 판정 제공
- 출발국·도착국 규정을 **교차 검색**하여 가장 엄격한 기준으로 판정
- 슬롯 필링 방식으로 **노선 → 물품 → 속성** 순서로 대화 흐름 관리
- **Dynamic Override**: 대화 도중 출발/도착지를 실시간으로 자유롭게 수정 가능
- **응답 속도 약 3.3배 향상**: 평균 50초 → 15초 이내 완결

---

## ⚙️ 기술적 도전과 해결

### 1. 검색 미적중 문제 — Data Augmentation Pipeline

**문제**: 사용자는 "보조배터리", "미숫가루" 같은 일상 단어를 쓰지만, 실제 규정 문서는 "리튬이온", "농산물/가공식품" 같은 공식 법률 용어로 작성되어 있어 벡터 검색 적중률이 낮았습니다.

**해결**: GPT-4o-mini를 활용해 84개 원본 문서마다 연관 동의어 및 예상 검색 키워드를 **문서당 15개 이상** 오프라인으로 증강 → Vector DB 검색 적중률을 대폭 끌어올렸습니다.

### 2. 응답 속도 문제 — Two-Track LLM + One-shot Prompting

**문제**: 슬롯 추출과 DB 매핑을 별도 API 호출로 처리하여 평균 응답 대기 시간이 **50초** 에 달했습니다.

**해결**:
- **One-shot Prompting**: 슬롯 추출과 DB 매핑 로직을 하나의 프롬프트로 통합하여 OpenAI API 호출 횟수를 2회 → 1회로 단축
- **reasoning_effort="low"**: 불필요한 내부 추론 루프를 제어하여 응답 품질 저하 없이 처리 속도 향상
- **결과**: 평균 응답 시간 50초 → **15초 이내** (약 **3.3배** 향상)

### 3. 환각(Hallucination) 방어

- 한국어로 질문해도 출발지를 한국으로 자동 추정하지 않고 `null`로 엄격 처리
- 미지원 노선(일본 등) 입력 시 **⚠️ 경고를 최우선으로 출력**하는 방어 로직 구현

---

## 🔁 RAG 파이프라인

```
[Offline] 원본 규정 문서 → Data Augmentation (GPT-4o-mini) → 임베딩 → ChromaDB

[Online]  사용자 질문
             ↓
         슬롯 추출 + DB 매핑 (One-shot, API 1회)
             ↓
         미완성 슬롯? → 핀포인트 질의
             ↓
         지원 노선 확인 → 미지원 시 ⚠️ 경고 선출력
             ↓
         메타데이터 필터 벡터 검색
             ↓
         문서 찾음? → RAG 답변 생성 (GPT-4o-mini)
                   → 못 찾음? → Fallback 답변 생성 (GPT-4o)
             ↓
         이모지 판정 + Bullet Point 가이드 출력
```

| 단계 | 이름 | 내용 |
|------|------|------|
| 0단계 | Data Augmentation | GPT-4o-mini로 규정 문서당 동의어/키워드 15개 이상 증강 (오프라인) |
| 1단계 | Data Ingestion | JSONL → text-embedding-3-small → ChromaDB |
| 2단계 | Router & Slot Filling | 노선·물품 슬롯 추출 및 DB 매핑 One-shot 통합 처리 |
| 3단계 | Rewriter & Retriever | 용어 정규화 + 메타데이터 필터 벡터 검색 + Fallback |
| 4단계 | Judge & Generator | 이모지 판정 + Bullet Point 형태 가이드 답변 생성 |

---

## 📁 프로젝트 구조

```
LANGCHAINTON_RAG/
├── app.py                              # Streamlit 챗봇 UI
├── bot_logic.py                        # RAG 파이프라인 핵심 로직
│                                       # (슬롯 필링, 검색, 판정, 생성)
├── data_augmenter.py                   # 원본 데이터 키워드/동의어 증강 스크립트
├── ingest.py                           # 데이터 임베딩 스크립트 (최초 1회 실행)
├── requirements.txt                    # 의존성 패키지 목록
│
├── data/
│   ├── index_docstore_export.jsonl     # 항공 규정 원문 데이터 (84개 문서)
│   └── index_docstore_augmented.jsonl  # 키워드 증강된 데이터셋
│
├── RAG_pipeline/
│   └── schema.md                       # RAG 파이프라인 설계 계획서
│
├── docs/
└── prompt/
```

---

## 🚀 빠른 시작

### 1. 가상환경 세팅

> **권장 환경**: Python 3.9.6

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. 환경변수 설정

프로젝트 루트에 `.env` 파일 생성 후 OpenAI API 키 입력:

```
OPENAI_API_KEY=sk-...
```

### 3. 데이터 증강 및 임베딩 (최초 1회)

```bash
# 원본 데이터에 키워드 증강 (약 1분 소요)
.venv/bin/python data_augmenter.py

# 증강된 데이터를 Vector DB에 주입
.venv/bin/python ingest.py
```

### 4. 앱 실행

```bash
.venv/bin/streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|---|---|
| LLM (기본 파이프라인) | GPT-4o-mini (`reasoning_effort` 튜닝) |
| LLM (Fallback) | GPT-4o |
| Embedding | text-embedding-3-small (OpenAI) |
| Vector DB | ChromaDB |
| Framework | LangChain |
| UI | Streamlit |

---

## 👥 팀원 소개

**팀명: LAGs (Liquids, Aerosols, and Gels)**

| 이름 | 역할 |
|---|---|
| 차병곤 | - |
| 손승희 | - |
| 상은영 | - |
| 이찬규 | RAG 파이프라인 구축 (슬롯 필링·검색·판정·생성), 프롬프트 엔지니어링, 발표 |
| 김선우 | - |

> 팀원 역할은 추후 업데이트 예정