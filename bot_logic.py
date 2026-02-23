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
embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
llm         = ChatOpenAI(model="gpt-5-mini", reasoning_effort="low")
advanced_llm = ChatOpenAI(model="gpt-5.2", reasoning_effort="low")
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)


# ─────────────────────────────────────────────────────────────
# DB 항목 목록 로드 (앱 시작 시 1회)
# ─────────────────────────────────────────────────────────────

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

COMBINED_SYSTEM_PROMPT = """당신은 항공 규정 챗봇 전문가입니다.
사용자 메시지와 대화 기록을 분석하여 다음 JSON 구조로만 정확히 출력하세요:
{
  "slots": {
    "departure": "출발국코드(KR/US/JP 등. 사용자가 새로운 출발지를 말하면 기존 값을 무시하고 새 출발국코드를 출력. 대화에 명시적으로 언급되지 않았다면 사용자의 언어나 정황으로 절대 유추하지 말고 반드시 null로 표기할 것)",
    "arrival": "도착국코드(사용자가 새로운 도착지를 말하면 기존 값을 무시하고 새 도착국코드를 출력. 대화에 명시되지 않았다면 반드시 null)",
    "item": "물품명(추출된 물품명 단 하나. 모르면 null)",
    "quantity": "수량/용량(모르면 null)"
  },
  "mapped_db_items": {
    "KR": ["출발/도착국 중 KR이 포함될 경우 아래 [KR DB 목록] 중 물품명과 관련된 항목 최대 3개 배열(없으면 [])"],
    "US": ["출발/도착국 중 US가 포함될 경우 아래 [US DB 목록] 중 물품명과 관련된 항목 최대 3개 배열(없으면 [])"]
  }
}
오직 순수 JSON 데이터만 출력하세요. 마크다운(```json)이나 다른 설명은 절대 추가하지 마세요.
"""

def extract_slots_and_map(user_message: str, chat_history: list[dict], current_slots: dict) -> tuple[dict, dict[str, list[str]]]:
    """대화 메시지에서 슬롯을 추출하고 DB 항목에 대한 매핑까지 한 번의 프롬프트로 처리."""
    history_text = ""
    for msg in chat_history[-6:]:
        role = "사용자" if msg["role"] == "user" else "봇"
        history_text += f"{role}: {msg['content']}\n"

    kr_db_str = "\n".join(f"  - {it}" for it in DB_ITEMS.get("KR", []))
    us_db_str = "\n".join(f"  - {it}" for it in DB_ITEMS.get("US", []))

    prompt = f"""현재 슬롯 상태: {json.dumps(current_slots, ensure_ascii=False)}

최근 대화:
{history_text}
사용자 최신 메시지: {user_message}

[KR DB 목록]
{kr_db_str}

[US DB 목록]
{us_db_str}

위 정보를 바탕으로 슬롯과 매핑 항목(mapped_db_items)을 한 번에 JSON으로 추출하세요."""

    response = llm.invoke([
        SystemMessage(content=COMBINED_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])
        parsed = json.loads(raw)
        
        new_slots = parsed.get("slots", {})
        mapped_items = parsed.get("mapped_db_items", {"KR": [], "US": []})
        
        merged = {**current_slots}
        for k, v in new_slots.items():
            if v is not None:
                merged[k] = v
                
        return merged, mapped_items
    except Exception:
        return current_slots, {"KR": [], "US": []}


def check_missing_slots(slots: dict) -> Optional[str]:
    """미확정 슬롯에 대한 재질문 문자열 반환. 모두 확정이면 None."""
    dep = slots.get("departure")
    arr = slots.get("arrival")
    item = slots.get("item")
    
    if not dep and not arr:
        return "✈️ 어디에서 출발하여 어디로 가시나요? (예: 한국 → 미국)"
    elif not dep:
        return "🛫 출발하시는 국가(또는 공항)를 알려주세요."
    elif not arr:
        return "🛬 도착하시는 국가(또는 공항)를 알려주세요."
        
    if not item:
        return "🎒 어떤 물건의 반입 규정이 궁금하신가요?"
        
    if dep == arr:
        return "⚠️ 출발지와 도착지가 같습니다. 다시 입력해 주세요."
        
    return None


def retrieve_docs(slots: dict, mapped: dict[str, list[str]]) -> tuple[list[dict], bool]:
    """
    확정된 슬롯과 매핑 결과로 ChromaDB에서 관련 문서 검색.
    [v3] 매핑 전체 실패 여부를 두 번째 반환값으로 제공.

    Returns:
        (retrieved_docs, all_mapping_failed)
        - all_mapping_failed=True  → 모든 jurisdiction에서 매핑 실패 → v3 일반지식 Fallback 필요
    """
    item          = slots.get("item", "")
    departure     = slots.get("departure", "KR")
    arrival       = slots.get("arrival", "US")
    jurisdictions = list({departure, arrival})

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
1. 답변에서 마크다운 형태의 볼드체(**), 기울임체(*) 등 별표(*) 기호를 **절대** 사용하지 마세요. 오직 평문 문자열과 하이픈(-) 기호만 사용합니다.
2. 출력 형식은 아래 [출력 예시]의 구조를 반드시 따르세요.
3. 첫 줄: 이모지(🟢/🟡/🔴)와 '기내/위탁' 가능 여부 한 줄 요약.
4. 중간 줄: 하이픈(-)을 사용해 항목별 상세 설명 (배터리 내장 여부 등 특이사항 분리).
5. 띄어쓰기를 철저히 지키고, 문단(의미)이 바뀔 때는 반드시 줄바꿈(엔터 2번)을 해서 널찍하고 읽기 쉽게 작성하세요.
6. 마지막 단락: 출처를 아래 형식의 공식 HTML 하이퍼링크로 제공하세요 (단순 텍스트나 마크다운 링크 출력 불가).
   - 한국(KR) 규정 참조 시: <a href="https://www.avsec365.or.kr/" target="_blank">항공보안365</a>
   - 미국(US) 규정 참조 시: <a href="https://www.cbp.gov/travel/us-citizens/know-before-you-go/prohibited-and-restricted-items" target="_blank">미국 관세국경보호청(CBP)</a>

[출력 예시]
🟢 기내·위탁 모두 가능(일반 제품) — 단, 특수 상황이면 제한될 수 있어요.

- 일반 제품: 규정 적용 없이 기내 반입·위탁 수하물 모두 통상 허용.
- 특정 조건(예: 배터리 내장형)이라면: 리튬배터리 규정 적용 → 보통 기내 반입 권장/위탁 제한 가능이라 항공사 확인 권장.

출처 확인: <a href="https://www.avsec365.or.kr/" target="_blank">항공보안365</a>
"""

GENERAL_KNOWLEDGE_SYSTEM_PROMPT = """친절한 항공 규정 안내원입니다. DB에 없지만 일반 규정을 추론해 쉽고 가독성 좋게 답변하세요.

규칙:
1. 답변에서 마크다운 형태의 볼드체(**), 기울임체(*) 등 별표(*) 기호를 **절대** 사용하지 마세요. 오직 평문 문자열과 하이픈(-) 기호만 사용합니다.
2. 출력 형식은 아래 [출력 예시]의 구조를 반드시 따르세요.
3. 첫 줄: 이모지(🟢/🟡/🔴)와 '기내/위탁' 가능 여부 한 줄 요약.
4. 중간 줄: 하이픈(-)을 사용해 항목별 상세 설명 (배터리 내장 여부 등 특이사항 분리).
5. 띄어쓰기를 철저히 지키고, 문단이 바뀔 때는 반드시 줄바꿈(엔터 2번)을 해서 널찍하고 읽기 쉽게 작성하세요.

[출력 예시]
🟢 기내·위탁 모두 가능(일반 전기 소형가전) — 단, 배터리 내장형/가스충전식이면 제한될 수 있어요.

- 일반 콘센트형 제품(배터리 없음): IATA 일반 기준으로 기내 반입·위탁 수하물 모두 통상 허용.
- 배터리(리튬이온) 내장형 무선 제품이라면: 리튬배터리 규정 적용 → 보통 기내 반입 권장/위탁 제한 가능(배터리 용량 확인 필요)이라 항공사 확인 권장.

⚠️ 정확한 규정은 이용 항공사 또는 <a href="https://www.avsec365.or.kr" target="_blank">항공보안365</a>에서 확인하세요."""

FALLBACK_MSG = (
    "😓 죄송합니다. 해당 물품에 대한 규정 정보를 데이터베이스에서 찾지 못했습니다.\n\n"
    "정확한 정보를 위해 이용하실 항공사 고객센터 또는 "
    "<a href='https://www.avsec365.or.kr' target='_blank'>항공보안365</a>를 통해 확인해 주세요."
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

    # 2단계: 슬롯 추출 및 DB 매핑 동시 처리
    t0 = time.time()
    updated_slots, mapped_items = extract_slots_and_map(user_message, chat_history, slots)
    t1 = time.time()
    print(f"⏱️ [1] 슬롯 추출 및 매핑 소요 시간: {t1 - t0:.2f}초")

    # 슬롯 미확정 시 재질문
    missing_q = check_missing_slots(updated_slots)
    if missing_q:
        return stream_string(missing_q), updated_slots

    # 3단계: 검색
    t2 = time.time()
    retrieved, all_mapping_failed = retrieve_docs(updated_slots, mapped_items)
    t3 = time.time()
    print(f"⏱️ [2] 벡터 DB 검색 소요 시간: {t3 - t2:.2f}초")

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
        
        # [v4] 타국가 등 미지원 노선에 대한 경고문 선출력
        supported = {"KR", "US"}
        dep = updated_slots.get("departure")
        arr = updated_slots.get("arrival")
        
        if (dep and dep not in supported) or (arr and arr not in supported):
            yield "⚠️ 현재 기내뭐돼 서비스는 한국(KR)과 미국(US) 노선 정밀 규정만 지원합니다. 타 국가 노선은 아래 안내와 다를 수 있으니 주의해 주세요.\n\n"

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
            "desc": "insight 1, 2: 물품만 있는 질문 (출발/도착지 묻는지 확인)",
            "message": "고추장 기내에 들고가도 돼?",
            "slots": {},
        },
        {
            "desc": "insight 3: 물품과 상세 항목이 있는 질문 (출발/도착지 묻는지 확인)",
            "message": "100Wh 보조배터 기내 반입 가능?",
            "slots": {},
        },
        {
            "desc": "insight 4: 타국가(일본) 질문에 대한 처리",
            "message": "나 한국에서 일본 가는데 액체류 돼?",
            "slots": {},
        },
        {
            "desc": "v4: 경로 중간 변경 처리 (한국->미국 대화 중 한국->중국으로 바꿈)",
            "message": "아 미안 나 미국 아니고 중국 가는데 그래도 똑같아?",
            "slots": {"departure": "KR", "arrival": "US", "item": "고추장"},
        },
        {
            "desc": "v4: 경로 중간 변경 처리 (한국->미국 대화 중 중국->캐나다로 전면 교체)",
            "message": "중국에서 캐나다로 갈 때는 초콜릿 어떻게 해야해?",
            "slots": {"departure": "KR", "arrival": "US", "item": "보조배터리"},
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
