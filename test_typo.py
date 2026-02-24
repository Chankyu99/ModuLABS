import asyncio
from bot_logic import extract_slots_and_map

async def test_typos():
    test_cases = [
        "한국에서 미국 가는데 보조베터리 가져가도 돼?",  # 보조베터리 (typo for 보조배터리)
        "시카고 가는데 고쥬장 반입 돼?",               # 고쥬장 (typo for 고추장)
        "일본 가는데 놋북 기내 반입 가능?",             # 놋북 (typo/abbr for 노트북)
        "미국갈때 블투 이어푠 챙길수 있어?",           # 이어푠 (typo for 이어폰)
        "한국에서 나트랑 갈때 전다담배 기내 반입 가능?"  # 전다담배 (typo for 전자담배)
    ]
    for msg in test_cases:
        print(f"입력: {msg}")
        slots, mapped = extract_slots_and_map(msg, [], {})
        print(f"추출 슬롯: {slots}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_typos())
