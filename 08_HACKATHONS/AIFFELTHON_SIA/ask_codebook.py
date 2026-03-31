"""
GDELT / CAMEO 코드북에 질문하는 도구
사용법: python3 ask_codebook.py "EventCode 195가 정확히 무슨 뜻이야?"
"""
import sys
import os
from pathlib import Path
from google import genai

# ── .env에서 API Key 로드 ──
ENV_PATH = BASE / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().strip().splitlines():
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip().strip('"').strip("'")

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("❌ GEMINI_API_KEY가 없습니다. .env 파일을 확인하세요.")
    sys.exit(1)
MODEL = "gemini-2.5-flash-preview-04-17"

BASE = Path(__file__).parent
CODEBOOKS = [
    BASE / "CAMEO.Manual.1.1b3.pdf",
    BASE / "GDELT-Data_Format_Codebook.pdf",
]

SYSTEM_PROMPT = """당신은 GDELT와 CAMEO 코드 체계의 전문가입니다.
주어진 PDF 코드북을 참고하여 질문에 정확하게 답변해주세요.
- 코드 번호와 공식 명칭을 반드시 포함하세요
- 예시 이벤트가 있으면 함께 제시하세요
- 답변은 한국어로 해주세요
- 위성 촬영 관점에서의 활용 가능성도 언급해주세요"""


def ask(question: str):
    client = genai.Client(api_key=API_KEY)

    # PDF 파일 업로드
    uploaded = []
    for pdf_path in CODEBOOKS:
        if pdf_path.exists():
            f = client.files.upload(file=pdf_path)
            uploaded.append(f)
            print(f"✅ 업로드: {pdf_path.name}")

    # 질문 전송
    contents = []
    for f in uploaded:
        contents.append(
            genai.types.Part.from_uri(file_uri=f.uri, mime_type="application/pdf")
        )
    contents.append(question)

    print(f"\n🔍 질문: {question}\n")
    print("─" * 60)

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
        ),
    )
    print(response.text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python3 ask_codebook.py \"질문\"")
        print()
        print("예시:")
        print('  python3 ask_codebook.py "EventCode 194와 195의 차이는?"')
        print('  python3 ask_codebook.py "QuadClass 4에 해당하는 모든 Root Code를 설명해줘"')
        print('  python3 ask_codebook.py "AvgTone 컬럼의 정확한 계산 방법은?"')
        print('  python3 ask_codebook.py "ActorType 코드 중 군사 관련 코드를 모두 알려줘"')
        sys.exit(1)

    question = sys.argv[1]
    ask(question)
