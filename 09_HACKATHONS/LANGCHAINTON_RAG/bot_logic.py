"""

bot_logic.py : RAG 파이프라인 구현 모듈

실행 흐름은 다음과 같다.

1) 사용자 질문 입력 → 슬롯 추출 및 DB 항목 매핑
2) 필수 슬롯 누락 여부 체크 → 누락 시 사용자에게 재질문 
3) ChromaDB에서 문서 검색
4) 검색된 문서 + 슬롯 정보를 기반으로 최종 답변 생성
5) Streamlit에 스트리밍 출력

"""

# 필요한 모듈 임포트

import json
# import os
# import time
# import concurrent.futures
from pathlib import Path
# from typing import Optional, Iterator

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

# API KEY 환경변수 로드
load_dotenv()

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
DATA_FILE = BASE_DIR / "data" / "index_docstore_export.jsonl"

# 검색 설정
COLLECTION_NAME = "airline_regulations"
RETRIEVAL_TOP_K = 5                     # 검색 결과 수
RETRIEVAL_SCORE_THRESHOLD = 1.2         # 검색 점수 임계값 (낮을수록 더 유사한 문서만 반환)
MAX_MAPPED_ITEMS = 3                    # LLM이 선택할 최대 DB 항목 수

# LLM 및 ChromaDB 초기화
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", max_tokens=2048, temperature=1.0)


vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)


# 1. 국가별 물품 카탈로그 생성 (추후 슬롯 매핑에 활용)

COUNTRIES = {"KR","US"} # 지원 국가 목록 (한국 -> 미국 시나리오 설정)

def load_country_item_catalog():
    """
    원본 JSONL(DATA_FILE)에서 국가별 공식 물품 카탈로그 생성
    
    반환 형식: {"KR": ["액체·분무·겔류"], "US": ["농산물/식품"]}
    """
    
    # 카탈로그 초기화
    items_by_country: dict[str, set[str]] = {country: set() for country in COUNTRIES}

    # JSONL 파일 읽고 국가별 물품 추출
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped_line = line.strip()

            if not stripped_line:
                continue
            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 디코딩 오류 (라인 {line_num}): {e}") from e
            
            country = record.get("country")
            item = record.get("item")

            if country not in COUNTRIES:
                continue

            if not isinstance(item, str) or not item.strip():
                continue

            items_by_country[country].add(item.strip())

    return {
        country: sorted(items)
        for country, items in items_by_country.items()
    }

# 모듈 로드 시 1회만 실행 -> 국가별 물품 카탈로그를 메모리에 저장해 매 질문마다 재사용
DB_ITEM_CATALOG : dict[str, list[str]] = load_country_item_catalog()


# 2. Router & Slot Filling  (슬롯 추출)

# LLM 시스템 프롬프트로 답변 방식 정의 (슬롯 추출 + DB 매핑을 한 번에 수행하도록 유도)

COMBINED_SYSTEM_PROMPT = """
You are an expert slot extraction and database item mapping engine
for an airline baggage regulation chatbot.

Your task is to analyze the user's latest message and recent chat history,
then return only a valid JSON object with the following structure:

{
  "slots": {
    "departure": "Departure country code such as KR, US, JP. If unknown, use null.",
    "arrival": "Arrival country code such as KR, US, JP. If unknown, use null.",
    "item": "One normalized item name. Correct Korean slang, abbreviations, and typos. If unknown, use null.",
    "quantity": "Quantity, capacity, volume, weight, or battery capacity such as 100ml, 2 items, 100Wh. If unknown, use null."
  },
  "mapped_db_items": {
    "KR": ["Up to 3 related KR database item names from the provided KR DB list. Use [] if none."],
    "US": ["Up to 3 related US database item names from the provided US DB list. Use [] if none."]
  }
}

Rules:
- Return only valid JSON.
- Do not use markdown code fences.
- Do not add explanations.
- The user message may be in Korean.
- Understand Korean slang, abbreviations, typos, and informal expressions.
- Normalize item names into standard Korean item names when possible.
- If the user clearly changes the route, prefer the newly mentioned departure or arrival.
- Do not infer a country from language alone.
- If a value is not explicitly mentioned or cannot be reliably inferred from chat history, use null.
- For mapped_db_items, choose only exact item names that appear in the provided DB lists.
"""

# 사용자 메시지 + 대화 히스토리 + 현재 슬롯 상태 → LLM 프롬프트 생성
def build_slot_prompt(user_message: str, chat_history: list[dict], current_slots: dict,) -> str:
    
    history_lines = []

    for message in chat_history[-6:]:
        role = "사용자" if message["role"] == "user" else "봇"
        history_lines.append(f"{role}: {message['content']}")

    history_text = "\n".join(history_lines)

    kr_db_text = "\n".join(f"  - {item}" for item in DB_ITEM_CATALOG.get("KR", []))
    us_db_text = "\n".join(f"  - {item}" for item in DB_ITEM_CATALOG.get("US", []))

    return f"""
            Current slot state:
            {json.dumps(current_slots, ensure_ascii=False)}

            Recent chat history:
            {history_text}

            User's latest message:
            {user_message}

            [KR DB List]
            {kr_db_text}

            [US DB List]
            {us_db_text}

            Return the extracted slots and mapped_db_items as valid JSON only.
            """


# LLM 응답 문자열을 딕셔너리로 바꾸고 파싱 → 슬롯 병합
def parse_slot_response(response_text: str, current_slots: dict,) -> tuple[dict, dict[str, list[str]]]:

    raw_text = response_text.strip()

    if raw_text.startswith("```"):
        raw_text = "\n".join(raw_text.split("\n")[1:-1])

    parsed = json.loads(raw_text)

    new_slots = parsed.get("slots", {})
    mapped_items = parsed.get("mapped_db_items", {"KR": [], "US": []})

    merged_slots = {**current_slots}

    for key, value in new_slots.items():
        if value is not None and value != "":
            merged_slots[key] = value

    return merged_slots, mapped_items


# 사용자 메시지로부터 슬롯 추출 → DB 매핑
def extract_slots_and_map(user_message: str, chat_history: list[dict], current_slots: dict,) -> tuple[dict, dict[str, list[str]]]:

    prompt = build_slot_prompt(user_message=user_message, 
                               chat_history=chat_history, 
                               current_slots=current_slots,)

    response = llm.invoke([
        SystemMessage(content=COMBINED_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    try:
        return parse_slot_response(
            response_text=chunk_to_text(response),
            current_slots=current_slots,
        )
    except Exception as error:
        print(f"[extract_slots_and_map] 슬롯 추출 실패: {error}")
        return current_slots, {"KR": [], "US": []}

# 3. 필수 슬롯 누락 여부 체크 -> 사용자의 첫 질문에서 출발지/도착지/물품 중 하나라도 빠졌다면 누락된 정보를 재질문하도록 설계
def check_missing_slots(slots: dict) -> str | None:

    departure = slots.get("departure")
    arrival = slots.get("arrival")
    item = slots.get("item")

    if not departure and not arrival:
        return "✈️ 어디에서 출발해서 어디로 가시나요? 예: 한국에서 미국"

    if not departure:
        return "🛫 출발 국가를 알려주세요."

    if not arrival:
        return "🛬 도착 국가를 알려주세요."

    if departure == arrival:
        return "⚠️ 출발지와 도착지가 같습니다. 다시 입력해 주세요."

    if not item:
        return "🎒 어떤 물건의 반입 규정이 궁금하신가요?"

    return None

# 4. ChromaDB에서 문서 검색

def retrieve_docs(slots: dict, mapped_items: dict[str, list[str]],) -> tuple[list[dict], bool]:

    """
    슬롯 정보 + 매핑된 DB 항목을 활용해 어떤 국가의 규정을 봐야할지 결정하고, 관련 문서를 검색해 리스트로 반환
    """

    item = slots.get("item", "")
    departure = slots.get("departure")
    arrival = slots.get("arrival")

    jurisdictions = []  # 검색할 관할 국가 코드 리스트 (프로젝트에서는 ["KR", "US"])

    for country_code in [departure, arrival]:
        if country_code and country_code not in jurisdictions: # 출발지와 도착지 중복 방지
            jurisdictions.append(country_code)
    
    # 매핑된 DB 항목이 있으면 해당 국가의 규정을 우선적으로 검색
    all_mapping_failed = all(
        len(mapped_items.get(country_code, [])) == 0
        for country_code in jurisdictions
    )

    retrieved_docs = []
    seen_doc_ids = set()    # 같은 문서가 여러 번 검색되는 것을 방지

    for country_code in jurisdictions:
        matched_items = mapped_items.get(country_code, [])

        if matched_items:
            query = " ".join(matched_items) + " " + item
        else:
            query = item

        search_results = vectorstore.similarity_search_with_score(
            query=query,
            k=RETRIEVAL_TOP_K,
            filter={"jurisdiction": country_code},
        )

        for doc, score in search_results:
            metadata = doc.metadata
            doc_id = metadata.get("doc_id", id(doc))
            db_item_name = metadata.get("item", "")

            if doc_id in seen_doc_ids:  # 이미 처리한 문서면 스킵
                continue

            if matched_items and db_item_name not in matched_items: # 매핑된 DB 항목이 있는데 검색 결과 문서의 항목이 매핑된 항목과 일치하지 않으면 스킵
                continue

            if not matched_items and score > RETRIEVAL_SCORE_THRESHOLD: # 검색 점수가 임계값보다 높으면 스킵 
                continue

            seen_doc_ids.add(doc_id)
            retrieved_docs.append({
                "doc": doc,
                "score": score,
                "jurisdiction": country_code,
                "mapped": bool(matched_items),
            })

    return retrieved_docs, all_mapping_failed

# 5. 검색된 문서 + 슬롯 정보를 기반으로 최종 답변 생성 -> app.py에서 스트리밍 출력과 함께 호출

# 최종 답변 생성을 위한 시스템 프롬프트
JUDGE_SYSTEM_PROMPT = """
You are a friendly airline baggage regulation assistant.

Use the retrieved regulation context to answer the user's question in Korean.

Rules:
- Answer in Korean.
- Do not use markdown bold, markdown tables, or code blocks.
- Start with one short summary line using one of these emojis:
  🟢 allowed, 🟡 conditional, 🔴 prohibited
- Then explain the details with short bullet points using hyphens.
- If the regulation differs between carry-on and checked baggage, explain both.
- If the user's item is a sub-item of a retrieved database category, naturally mention which category was used.
- Do not invent rules that are not supported by the provided context.
- End with official source links when relevant.

Source link rules:
- For KR regulations, include:
  <a href="https://www.avsec365.or.kr/" target="_blank">항공보안365</a>
- For US regulations, include:
  <a href="https://www.cbp.gov/travel/us-citizens/know-before-you-go/prohibited-and-restricted-items" target="_blank">미국 관세국경보호청(CBP)</a>
"""

# 검색된 문서 리스트를 LLM 프롬프트에 넣을 context 문자열로 변환
def build_retrieved_context(retrieved_docs: list[dict]) -> str:

    context_parts = []

    for result in retrieved_docs:
        # LangChain Document 객체에 맞게 doc과 metadata 추출
        doc = result["doc"]
        metadata = doc.metadata

        jurisdiction = metadata.get("jurisdiction", "?") # 있으면 가져오고 없으면 ?로 표시 
        stage = metadata.get("stage", "?")
        item = metadata.get("item", "?")

        context_parts.append(
            f"[{jurisdiction} regulation / {stage}]\n"
            f"Database item: {item}\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(context_parts) # 검색된 문서들을 구분자를 연결해 하나의 문자열로 반환

# 최종 답변 생성
def generate_answer(user_message: str, slots: dict, retrieved_docs: list[dict],):

    context = build_retrieved_context(retrieved_docs)

    departure = slots.get("departure", "?")
    arrival = slots.get("arrival", "?")
    item = slots.get("item", "?")
    quantity = slots.get("quantity", "")

    prompt = f"""
                Route:
                {departure} -> {arrival}

                User item:
                {item}

                Quantity or capacity:
                {quantity}

                User's original question:
                {user_message}

                Retrieved regulation context:
                {context}

                Answer the user's question based only on the retrieved regulation context.
                """

    # invoke()대신 사용해 답변 생성과 동시에 스트리밍 출력으로 UX 개선
    return llm.stream([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])


# 6. 검색 결과가 없을 때, 없는 지식을 지어내기 보다 항공 규정을 근거 없이 추측하지 않고 정확한 출처를 참고하라고 권유

NO_MAPPING_FALLBACK_MSG = (
    "😓 죄송합니다. 질문하신 물품과 직접 연결되는 규정 항목을 찾지 못했습니다.\n\n"
    "물품명을 더 구체적으로 입력해 보시거나, 항공사 또는 공식 규정 사이트에서 확인해 주세요."
)

NO_RETRIEVAL_FALLBACK_MSG = (
    "😓 죄송합니다. 관련 항목은 추정했지만, 신뢰할 만한 규정 문서를 찾지 못했습니다.\n\n"
    "정확한 확인을 위해 항공사 또는 공식 규정 사이트를 확인해 주세요."
)

# 7. Streamlit 출력용 반복가능한 스트림 처리 helper 함수 

def stream_string(message: str):
    """
    일반 문자열도 LLM 스트리밍 응답처럼 for문으로 순회할 수 있게 만든다.
    """

    yield message


def chunk_to_text(chunk) -> str:

    # LangChain의 AIMessageChunk는 보통 chunk.content에 실제 텍스트가 들어 있어 hasattr로 content 속성 존재 여부를 체크
    if hasattr(chunk, "content"):
        content = chunk.content
    else:
        content = chunk

    # 출력할 텍스트가 없으면 빈 문자열로 처리
    if content is None:
        return ""

    # 이미 문자열이면 그대로 반환
    if isinstance(content, str):
        return content

    # 모델에 따라 content가 여러 조각이 담긴 list로 올 때
    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):
                if "text" in block:
                    text_parts.append(block["text"])
                elif "content" in block:
                    text_parts.append(str(block["content"]))

            else:
                # getattr로 obj.text가 있으면 가져오고, 없으면 None을 반환
                block_text = getattr(block, "text", None)

                if block_text:
                    text_parts.append(block_text)
                else:
                    text_parts.append(str(block))

        return "".join(text_parts)

    # 위 경우에 해당하지 않는 값은 마지막으로 문자열 변환해서 반환
    return str(content)

# 8. 파이프라인 함수 : 지금까지 만든 함수를 연결

def run_pipeline(user_message: str, chat_history: list[dict], slots: dict):
    """
    전체 RAG 파이프라인을 실행 (슬롯 추출 → 누락 체크 → 문서 검색 → 답변 생성)

    최종 반환값:
    - bot_response_stream: Streamlit에서 for문으로 출력할 수 있는 응답 스트림
    - updated_slots: 이번 질문을 반영한 최신 슬롯 상태
    """

    # 사용자 메시지에서 슬롯 추출 + DB 항목 매핑
    updated_slots, mapped_items = extract_slots_and_map(
        user_message=user_message,
        chat_history=chat_history,
        current_slots=slots,
    )

    # 출발지/도착지/물품 중 빠진 값이 있으면 검색하지 않고 재질문
    missing_question = check_missing_slots(updated_slots)

    if missing_question:
        return stream_string(missing_question), updated_slots

    # 슬롯이 충분하면 ChromaDB에서 관련 규정 문서 검색
    retrieved_docs, all_mapping_failed = retrieve_docs(
        slots=updated_slots,
        mapped_items=mapped_items,
    )

    # 검색 결과에 따라 답변 스트림 결정
    if retrieved_docs:
        raw_stream = generate_answer(
            user_message=user_message,
            slots=updated_slots,
            retrieved_docs=retrieved_docs,
        )

    elif all_mapping_failed:
        raw_stream = stream_string(NO_MAPPING_FALLBACK_MSG)

    else:
        raw_stream = stream_string(NO_RETRIEVAL_FALLBACK_MSG)

    # LLM chunk 또는 fallback 문자열을 모두 순수 문자열 stream으로 변환
    def response_stream():
        for chunk in raw_stream:
            text = chunk_to_text(chunk)

            if text:
                yield text

    return response_stream(), updated_slots