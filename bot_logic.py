"""
bot_logic.py
------------
schema.md 2~4단계 RAG 파이프라인 구현.

  2단계: Router & Slot Filling  — 대화에서 {출발지, 도착지, 물품} 추출
  3단계: Rewriter & Retriever   — DB 목록 기반 항목 매핑 + 메타데이터 필터 벡터 검색
  4단계: Judge & Generator      — 판정(🟢/🟡/🔴) + Bullet Point 답변 생성

[v2 개선]
  - normalize_item() → map_item_to_db() 로 교체
    : LLM이 자유 생성하던 방식 → DB 84개 항목 목록 직접 참조 후 선택
    : "칼" → DB에서 "날 길이 6cm 초과 칼", "도끼·손도끼·큰 식칼 등 절단용 칼" 매핑
    : "미숫가루" → DB에서 US "가공/캔 식품" 카테고리 매핑

[v3 개선]
  - DB 매핑 전체 실패 시 단순 Fallback → LLM 일반 지식 기반 답변으로 격상
    : LLM이 물품을 추론 → 유사 카테고리 판단 → 항공 전문 지식으로 답변
    : 답변 하단에 ⚠️ "DB 미등재 항목, 일반 규정 기반 안내" 단서 명시
    : "보조배터리", "드라이기", "뜨개바늘" 등 DB 없는 물품 처리 가능

단독 테스트:
    .venv/bin/python bot_logic.py
"""

import json
import os
import time
import concurrent.futures
from pathlib import Path
from typing import Optional, Iterator

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ── 경로 / 상수 ────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
CHROMA_DIR      = BASE_DIR / "chroma_db"
DATA_FILE       = BASE_DIR / "data" / "index_docstore_export.jsonl"
COLLECTION_NAME = "airline_regulations"
TOP_K           = 5          # 검색 결과 수
MAX_MAPPED      = 3          # LLM이 선택할 최대 DB 항목 수

# ── 모델 초기화 ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_models():
    """LLM 및 벡터스토어 로드를 싱글톤으로 관리 (Streamlit 권장)"""
    _embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
    _llm         = ChatOpenAI(model="gpt-5-mini", temperature=0)
    _advanced_llm = ChatOpenAI(model="gpt-5.2", temperature=0.2)
    _vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return _llm, _advanced_llm, _vectorstore

llm, advanced_llm, vectorstore = get_models()


# ─────────────────────────────────────────────────────────────
# DB 항목 목록 로드 (앱 시작 시 1회)
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_db_items() -> dict[str, list[str]]:
    """
    JSONL에서 국가별 item 목록을 로드.
    반환: {"KR": [...], "US": [...]}
    """
    items: dict[str, list[str]] = {}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            country = rec.get("country", "?")
            item    = rec.get("item", "")
            if country not in items:
                items[country] = []
            if item and item not in items[country]:
                items[country].append(item)
    return items


# 모듈 로드 시 1회만 실행
DB_ITEMS: dict[str, list[str]] = load_db_items()


# ─────────────────────────────────────────────────────────────
# 2단계: 슬롯 추출 (Router & Slot Filling)
# ─────────────────────────────────────────────────────────────

SLOT_SYSTEM_PROMPT = """당신은 항공 규정 챗봇입니다.
사용자 메시지에서 다음 4가지 슬롯을 순수 JSON으로 추출하세요:
{"departure":"출발국코드(KR/US/JP 등)","arrival":"도착국코드","item":"물품명","quantity":"수량/용량"}
모르면 null, 출발/도착이 같아도 추출."""

@st.cache_data(show_spinner=False, ttl=3600)
def extract_slots(user_message: str, chat_history: list[dict], current_slots: dict) -> dict:
    """대화 메시지에서 슬롯(출발지, 도착지, 물품, 속성)을 추출."""
    history_text = ""
    for msg in chat_history[-6:]:
        role = "사용자" if msg["role"] == "user" else "봇"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""현재 슬롯 상태: {json.dumps(current_slots, ensure_ascii=False)}

최근 대화:
{history_text}
사용자 최신 메시지: {user_message}

위 정보를 바탕으로 슬롯을 추출하세요. 기존에 확정된 슬롯은 유지하세요."""

    response = llm.invoke([
        SystemMessage(content=SLOT_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    try:
        new_slots = json.loads(response.content.strip())
        merged = {**current_slots}
        for k, v in new_slots.items():
            if v is not None:
                merged[k] = v
        return merged
    except json.JSONDecodeError:
        return current_slots


def check_missing_slots(slots: dict) -> Optional[str]:
    """미확정 슬롯에 대한 재질문 문자열 반환. 모두 확정이면 None."""
    if not slots.get("departure") or not slots.get("arrival"):
        return "✈️ 어디에서 출발하여 어디로 가시나요? (예: 한국 → 미국)"
    if not slots.get("item"):
        return "🎒 어떤 물건의 반입 규정이 궁금하신가요?"
    if slots.get("departure") == slots.get("arrival"):
        return "⚠️ 출발지와 도착지가 같습니다. 다시 입력해 주세요."
    return None


MAP_SYSTEM_PROMPT = """DB 매핑 전문가입니다.
사용자 물품과 가장 관련된 DB 항목을 최대 {max_mapped}개 골라 JSON 배열로 출력하세요.
없으면 [] 출력."""

MAPPING_CACHE: dict[tuple[str, str], list[str]] = {}

def map_item_to_db(item: str, jurisdictions: list[str]) -> dict[str, list[str]]:
    """
    [v2 핵심] 사용자 물품명 → DB 항목 목록에서 관련 항목 선택.
    각 jurisdiction(KR/US)별로 관련 DB 항목을 선택하여 반환.
    반환: {"KR": ["날 길이 6cm 초과 칼", ...], "US": [...]}
    """
    result: dict[str, list[str]] = {}
    item_clean = item.strip()

    def fetch_for_jur(jur: str) -> tuple[str, list[str]]:
        db_list = DB_ITEMS.get(jur, [])
        if not db_list or not item_clean:
            return jur, []

        # 🚀 [초고속 최적화 1] 이전 검색 캐시 히트 (LLM 파싱 생략)
        cache_key = (item_clean, jur)
        if cache_key in MAPPING_CACHE:
            return jur, MAPPING_CACHE[cache_key]

        # 🚀 [초고속 최적화 2] 사용자가 DB 카테고리명과 완전히 똑같이 쳤으면 LLM 파싱 생략
        if item_clean in db_list:
            MAPPING_CACHE[cache_key] = [item_clean]
            return jur, [item_clean]

        db_list_str = "\n".join(f"  - {it}" for it in db_list)
        prompt = f"""사용자 물품: "{item_clean}"\n\n[{jur}] DB 항목 목록:\n{db_list_str}\n\n위 DB 항목 중, 사용자 물품 "{item_clean}"과 관련된 항목을 골라주세요."""

        try:
            response = llm.invoke([
                SystemMessage(content=MAP_SYSTEM_PROMPT.format(max_mapped=MAX_MAPPED)),
                HumanMessage(content=prompt),
            ])
            # 코드블록 등 감싸진 경우 정리
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])
            mapped = json.loads(raw)
            # DB에 실제로 있는 항목만 필터
            valid = [m for m in mapped if m in db_list][:MAX_MAPPED]
            
            # 캐시 저장
            MAPPING_CACHE[cache_key] = valid
            return jur, valid
        except (json.JSONDecodeError, TypeError, Exception):
            return jur, []

    # 병렬 처리로 각 국가별 매핑 응답 소요 시간 대폭 단축
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_for_jur, jur): jur for jur in jurisdictions}
        for future in concurrent.futures.as_completed(futures):
            jur, valid = future.result()
            result[jur] = valid

    return result


def retrieve_docs(slots: dict) -> tuple[list[dict], bool]:
    """
    확정된 슬롯으로 ChromaDB에서 관련 문서 검색.
    [v2] DB 목록 매핑 결과로 검색 쿼리를 구성.
    [v3] 매핑 전체 실패 여부를 두 번째 반환값으로 제공.

    Returns:
        (retrieved_docs, all_mapping_failed)
        - all_mapping_failed=True  → 모든 jurisdiction에서 매핑 실패 → v3 일반지식 Fallback 필요
    """
    item          = slots.get("item", "")
    departure     = slots.get("departure", "KR")
    arrival       = slots.get("arrival", "US")
    jurisdictions = list({departure, arrival})

    # DB 항목 매핑
    mapped = map_item_to_db(item, jurisdictions)
    print(f"[retrieve_docs] mapped: {mapped}")  # 디버그 로그

    # 모든 jurisdiction에서 매핑이 비었는지 체크
    all_mapping_failed = all(len(mapped.get(jur, [])) == 0 for jur in jurisdictions)

    all_docs = []
    seen_ids = set()

    for jur in jurisdictions:
        matched_items = mapped.get(jur, [])

        if matched_items:
            query = " ".join(matched_items) + " " + item
        else:
            query = item

        results = vectorstore.similarity_search_with_score(
            query=query,
            k=TOP_K,
            filter={"jurisdiction": jur},
        )

        for doc, score in results:
            doc_id       = doc.metadata.get("doc_id", id(doc))
            db_item_name = doc.metadata.get("item", "")

            if matched_items:
                if db_item_name in matched_items:
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append({
                            "doc": doc, "score": score,
                            "jurisdiction": jur, "mapped": True
                        })
            else:
                if score <= 1.2 and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append({
                        "doc": doc, "score": score,
                        "jurisdiction": jur, "mapped": False
                    })

    return all_docs, all_mapping_failed


# ─────────────────────────────────────────────────────────────
# 4단계: 최종 판정 + 답변 생성
# ─────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """친절한 항공 규정 안내원입니다. 가장 쉽고 가독성 좋게 답변하세요.

규칙:
1. 첫 단락: 이모지(🟢/🟡/🔴)와 핵심 판정 결과 요약. (출/도착국 중 하나라도 금지면 🔴 반입 불가)
2. 필수 포함: '기내 반입'과 '위탁 수하물' 각각의 가능 여부를 명확히 밝힐 것.
3. 이어지는 단락: 허용 용량/수량, 특이사항 등을 쉬운 용어로 설명.
4. 띄어쓰기를 철저히 지키고, 문단(의미)이 바뀔 때는 반드시 줄바꿈(엔터 2번)을 해서 널찍하고 읽기 쉽게 작성할 것.
5. 기계적인 DB 카테고리 매핑 설명 등 불필요한 사족은 철저히 생략. 모호할 경우 항공사 문의 권장 추가.
6. 마지막 단락: 출처를 아래 형식의 공식 하이퍼링크로 제공할 것 (단순 텍스트 메타데이터 출력 불가).
   - 한국(KR) 규정 참조 시: [항공보안365](https://www.avsec365.or.kr/)
   - 미국(US) 규정 참조 시: [미국 관세국경보호청(CBP)](https://www.cbp.gov/travel/us-citizens/know-before-you-go/prohibited-and-restricted-items)"""

GENERAL_KNOWLEDGE_SYSTEM_PROMPT = """친절한 항공 규정 안내원입니다. DB에 없지만 일반 규정을 추론해 쉽고 가독성 좋게 답변하세요.

규칙:
1. 첫 단락: 이모지(🟢/🟡/🔴)와 일반 규정에 따른 핵심 판정 결과 요약.
2. 필수 포함: '기내 반입'과 '위탁 수하물' 각각의 가능 여부를 명확히 밝힐 것.
3. 이어지는 단락: 핵심 주의사항(조건/수량 등)을 쉬운 용어로 설명.
4. 띄어쓰기를 철저히 지키고, 문단이 바뀔 때는 반드시 줄바꿈(엔터 2번)을 해서 널찍하고 읽기 쉽게 작성할 것.
5. 기계적인 분석 나열은 생략. 확신할 수 없으면 항공사 확인을 권장.

마지막 단락 (반드시 줄바꿈 후 아래 문구 그대로 포함):
⚠️ 일반 국제 규정 안내입니다. 탑승 전 탑승하시는 항공사에 꼭 확인해 주세요!"""

FALLBACK_MSG = (
    "😓 죄송합니다. 해당 물품에 대한 규정 정보를 데이터베이스에서 찾지 못했습니다.\n\n"
    "정확한 정보를 위해 이용하실 **항공사 고객센터** 또는 "
    "**[항공보안365](https://www.avsec365.or.kr)**를 통해 확인해 주세요."
)


def generate_answer(user_message: str, slots: dict, retrieved: list[dict]):
    """[DB 매핑 성공] 검색 결과 기반 최종 답변 생성 (스트리밍)."""
    context_parts = []
    for r in retrieved:
        doc  = r["doc"]
        meta = doc.metadata
        context_parts.append(
            f"[{meta.get('jurisdiction', '?')} 규정 / {meta.get('stage', '?')}]\n"
            f"항목: {meta.get('item', '?')}\n"
            f"{doc.page_content}"
        )
    context = "\n\n".join(context_parts)

    departure = slots.get("departure", "?")
    arrival   = slots.get("arrival", "?")
    item      = slots.get("item", "?")

    prompt = f"""노선: {departure} → {arrival}
사용자가 물어본 물품: {item}
사용자 질문: {user_message}

검색된 규정:
{context}

위 규정을 바탕으로 답변해주세요.
만약 사용자 물품이 DB 항목의 하위 개념이라면(예: '칼' → '날 길이 6cm 초과 칼'), 어떤 규정을 참조했는지 자연스럽게 안내해주세요."""

    return llm.stream([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])


def general_knowledge_answer(user_message: str, slots: dict):
    """[v3] DB 매핑 전체 실패 시 LLM 일반 항공 지식으로 추론·답변 (스트리밍)."""
    departure = slots.get("departure", "?")
    arrival   = slots.get("arrival", "?")
    item      = slots.get("item", "?")

    prompt = f"""노선: {departure} → {arrival}
사용자가 물어본 물품: "{item}"
사용자 질문: {user_message}
위 조건에 맞는 답변 부탁해."""

    return advanced_llm.stream([
        SystemMessage(content=GENERAL_KNOWLEDGE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])


# ─────────────────────────────────────────────────────────────
# 전체 파이프라인 진입점
# ─────────────────────────────────────────────────────────────

def stream_string(s: str) -> Iterator[str]:
    yield s

def run_pipeline(
    user_message: str,
    chat_history: list[dict],
    slots: dict,
) -> tuple[Iterator[str], dict]:
    """RAG 파이프라인 실행. Returns: (bot_response_stream, updated_slots)"""
    
    start_time = time.time()

    # 포괄적 질문 감지
    broad_keywords = ["다 알려", "전부", "모두", "목록", "리스트"]
    if any(kw in user_message for kw in broad_keywords) and not slots.get("item"):
        return stream_string(
            "🗂️ 어떤 카테고리의 규정이 궁금하신가요?\n\n"
            "아래 중 하나를 선택하거나, 직접 물품명을 입력해 주세요.\n"
            "- 🔫 총기·무기류\n"
            "- 🔪 도검·공구류\n"
            "- 💊 의약품·의료기기\n"
            "- 🧴 액체·겔·분무류\n"
            "- 🔋 배터리·전자기기\n"
            "- 🍎 식품·농산물\n"
            "- 💰 현금·귀중품"
        ), slots

    # 2단계: 슬롯 추출
    t0 = time.time()
    updated_slots = extract_slots(user_message, chat_history, slots)
    t1 = time.time()
    print(f"⏱️ [1] 슬롯 추출 소요 시간: {t1 - t0:.2f}초")

    # 슬롯 미확정 시 재질문
    missing_q = check_missing_slots(updated_slots)
    if missing_q:
        return stream_string(missing_q), updated_slots

    # 3단계: DB 매핑 + 검색
    t2 = time.time()
    retrieved, all_mapping_failed = retrieve_docs(updated_slots)
    t3 = time.time()
    print(f"⏱️ [2] DB 매핑 및 검색 소요 시간: {t3 - t2:.2f}초")

    # 4단계: 답변 생성 (스트리밍 연결용)
    if retrieved:
        # DB에서 문서를 찾은 경우 → 정규 RAG 답변
        raw_stream = generate_answer(user_message, updated_slots, retrieved)
    elif all_mapping_failed:
        # [v3] DB 매핑 자체가 전혀 안 된 경우 → LLM 일반 지식 답변
        print(f"[run_pipeline] DB 매핑 실패 → 일반 지식 Fallback: {updated_slots.get('item')}")
        raw_stream = general_knowledge_answer(user_message, updated_slots)
    else:
        # 매핑은 됐으나 score 초과로 문서 없음 → 기존 Fallback
        raw_stream = stream_string(FALLBACK_MSG)

    def traced_stream() -> Iterator[str]:
        t4 = time.time()
        for chunk in raw_stream:
            # chunk.content가 있는 경우(AIMessageChunk)와 바로 문자열(generator)인 경우 모두 처리
            if hasattr(chunk, "content"):
                yield chunk.content
            else:
                yield chunk
        t5 = time.time()
        print(f"⏱️ [3] 최종 답변 생성(스트리밍 완료) 소요 시간: {t5 - t4:.2f}초")
        print(f"⏱️ [Total] 전체 파이프라인 소요 시간: {t5 - start_time:.2f}초")

    return traced_stream(), updated_slots


# ─────────────────────────────────────────────────────────────
# 단독 테스트
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🛫 기내뭐돼 v3 — 일반지식 Fallback 테스트")
    print("=" * 60)

    test_cases = [
        {
            "desc": "v3 신규: 보조배터리 (DB 미등재)",
            "message": "한국→미국 보조배터리 기내 반입 가능해?",
            "slots": {},
        },
        {
            "desc": "v3 신규: 캐시 테스트 (보조배터리 재검색, 매핑 0초 확인)",
            "message": "보조배터리",
            "slots": {"departure": "KR", "arrival": "US"},
        },
        {
            "desc": "v3 신규: 드라이기 (DB 미등재)",
            "message": "드라이기는 가져갈 수 있어?",
            "slots": {"departure": "KR", "arrival": "US"},
        },
        {
            "desc": "v2 유지: 칼 (DB 간접 매핑)",
            "message": "칼은?",
            "slots": {"departure": "KR", "arrival": "US"},
        },
        {
            "desc": "v2 유지: 미숫가루 (US 식품 매핑)",
            "message": "미숫가루는?",
            "slots": {"departure": "KR", "arrival": "US"},
        },
    ]

    for i, tc in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] {tc['desc']}")
        print(f"  입력: {tc['message']}")
        response_stream, new_slots = run_pipeline(tc["message"], [], tc["slots"])
        print(f"  → 슬롯: {new_slots}")
        print("  → 응답: ", end="")
        for chunk in response_stream:
            print(chunk, end="", flush=True)
        print("\n" + "-" * 60)
