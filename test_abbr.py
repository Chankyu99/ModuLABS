import asyncio
from bot_logic import extract_slots_and_map

async def test_abbreviations():
    test_cases = [
        "한국에서 미국 가는데 보배 가져가도 돼?",
        "시카고 가는데 아아 반입 돼?",
        "일본 가는데 놋북 기내 반입 가능?",
        "미국갈때 블투 이어폰 챙길수 있어?",
        "한국에서 나트랑 갈때 전담 기내 반입 가능?"
    ]
    for msg in test_cases:
        print(f"입력: {msg}")
        slots, mapped = extract_slots_and_map(msg, [], {})
        print(f"추출 슬롯: {slots}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_abbreviations())
