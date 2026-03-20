import time
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from bot_logic import DB_ITEMS, extract_slots, map_item_to_db

load_dotenv()

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

COMBINED_SYSTEM_PROMPT = """당신은 항공 규정 챗봇 전문가입니다.
사용자 메시지와 대화 기록을 분석하여 다음 JSON 구조로만 정확히 출력하세요:
{
  "slots": {
    "departure": "출발국코드(KR/US/JP 등. 모르면 null)",
    "arrival": "도착국코드(모르면 null)",
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

def test_current_method(user_message: str):
    print("="*50)
    print("▶️ [기존 방식] 분리 호출 (Slot 추출 -> DB 매핑 병렬)")
    
    # 1. Slot 추출
    t0 = time.time()
    slots = extract_slots(user_message, [], {})
    t1 = time.time()
    print(f"  - 슬롯 추출 소요 시간: {t1 - t0:.2f}초 | 추출됨: {slots.get('item', '')}")
    
    # 2. DB 매핑
    jurisdictions = [j for j in {slots.get('departure'), slots.get('arrival')} if j]
    mapped = map_item_to_db(slots.get('item', ''), jurisdictions)
    t2 = time.time()
    print(f"  - DB 매핑 병렬 소요 시간: {t2 - t1:.2f}초 | 매핑 결과: {mapped}")
    print(f"  --> 총 소요 시간: {t2 - t0:.2f}초")
    print("="*50)

def test_combined_method(user_message: str):
    print("="*50)
    print("▶️ [신규 제안 방식] 단일 호출 (Slot 추출 + DB 매핑 통합)")
    
    kr_db_str = "\n".join(f"  - {it}" for it in DB_ITEMS.get("KR", []))
    us_db_str = "\n".join(f"  - {it}" for it in DB_ITEMS.get("US", []))
    
    prompt = f"""사용자 메시지: "{user_message}"

[KR DB 목록]
{kr_db_str}

[US DB 목록]
{us_db_str}

위 [KR DB 목록] 및 [US DB 목록]을 참고하여 슬롯과 매핑 항목(mapped_db_items)을 한 번에 JSON으로 추출하세요."""

    t0 = time.time()
    
    response = llm.invoke([
        SystemMessage(content=COMBINED_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])
        
    try:
        parsed = json.loads(raw)
    except Exception as e:
        parsed = f"JSON 파싱 에러: {e}\n원본 출력:\n{raw}"
        
    t1 = time.time()
    print(f"  - 단일 프롬프트 소요 시간: {t1 - t0:.2f}초")
    print(f"  - 결과값:\n{json.dumps(parsed, ensure_ascii=False, indent=2)}")
    print(f"  --> 총 소요 시간: {t1 - t0:.2f}초")
    print("="*50)

if __name__ == "__main__":
    test_msg = "한국에서 미국으로 보조배터리 가져갈 수 있어?"
    print(f"테스트 질문: {test_msg}\n")
    
    test_current_method(test_msg)
    test_combined_method(test_msg)
