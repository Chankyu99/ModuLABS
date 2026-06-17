"""

app.py : Streamlit 챗봇 UI 구성 모듈

"""

# 필요한 라이브러리 임포트
import streamlit as st
from bot_logic import run_pipeline

# 상수 및 설정값 모음

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
}

/* 전체 배경 */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* 헤더 영역 */
.hero-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a8edea, #fed6e3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    color: #b0b8d1;
    font-size: 1rem;
    margin-top: 0;
}

/* 슬롯 상태 배지 */
.slot-badge {
    display: inline-block;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    color: #d0d8f0;
    margin: 2px;
    backdrop-filter: blur(6px);
}

/* 채팅 버블 */
.chat-user {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    margin: 8px 0 8px 20%;
    box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    line-height: 1.6;
}
.chat-bot {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    color: #e8eaf6;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    margin: 8px 20% 8px 0;
    backdrop-filter: blur(10px);
    line-height: 1.7;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.sender-label {
    font-size: 0.72rem;
    color: #8892b0;
    margin-bottom: 4px;
    font-weight: 500;
}

/* 입력창 */
.stChatInputContainer {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}

/* 사이드바 */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.85) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
section[data-testid="stSidebar"] * {
    color: #c8d0e8 !important;
}

/* 버튼 */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}
</style>
"""
LOADING_HTML = """
<style>
.loader {
  border: 4px solid #f3f3f3;
  border-top: 4px solid #3498db;
  border-radius: 50%;
  width: 16px;
  height: 16px;
  animation: spin 1s linear infinite;
  display: inline-block;
  vertical-align: middle;
  margin-right: 8px;
}
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>
<div class="chat-bot">
  <div class="sender-label">기내뭐돼 봇</div>
  <div style="display: flex; align-items: center; color: rgba(255,255,255,0.7);">
    <div class="loader"></div>
    🔍 항공 규정 검색 및 답변 작성 중...
  </div>
</div>
"""

# CSS 적용

def apply_custom_css() -> None:

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 세션 초기화
def initialize_session_state() -> None:
    
    # Streamlit 세션 상태에 messages와 slots를 준비
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "slots" not in st.session_state:
        st.session_state.slots = {}


# 말풍선 렌더링 함수

# 사용자 메시지를 오른쪽 말풍선으로 출력
def render_user_message(content: str) -> None:
    
    st.markdown(f"""
                <div class="chat-user">
                <div class="sender-label" style="color:rgba(255,255,255,0.6);">나</div>
                {content}
                </div>
                """, unsafe_allow_html=True)

# 봇 메시지를 왼쪽 말풍선으로 출력
def render_bot_message(content: str) -> None:

    content_html = content.replace("\n", "<br>")

    st.markdown(f"""
<div class="chat-bot">
  <div class="sender-label">기내뭐돼 봇</div>
  {content_html}
</div>
""", unsafe_allow_html=True)

# 헤더 함수 -> 메인 화면 상단 제목을 출력
def render_header() -> None:
    st.markdown("""
                <div class="hero-header">
                <div class="hero-title">✈️ 기내뭐돼</div>
                <p class="hero-subtitle">가져갈까 말까? 비행기 짐싸기 고민 해결! 🎒</p>
                </div>
                """, unsafe_allow_html=True)

# 초기 안내 메시지
def render_welcome_message() -> None:
    """대화가 없을 때 처음 보여줄 안내 메시지를 출력한다."""
    if st.session_state.messages:
        return

    st.markdown("""
                <div class="chat-bot">
                <div class="sender-label">기내뭐돼 봇</div>
                안녕하세요! 저는 항공 반입 규정 안내 챗봇 <b>기내뭐돼</b>입니다. 🛫<br><br>
                <b>출발지</b>와 <b>도착지</b>, 그리고 <b>물품</b>을 알려주시면 기내/위탁 반입 가능 여부를 안내해 드릴게요.<br><br>
                예시 질문:<br>
                • <i>"한국에서 미국 갈 때 고추장 가져갈 수 있어?"</i>
                </div>
                """, unsafe_allow_html=True)

# 사이드바 렌더링 함수 -> 현재 슬롯 상태와 초기화 버튼, 사용 팁을 출력
def render_sidebar() -> None:
    """사이드바에 현재 슬롯 상태와 사용 팁을 출력한다."""
    with st.sidebar:
        st.markdown("## ✈️ 기내뭐돼")
        st.markdown("항공 반입 규정 RAG 챗봇")
        st.divider()

        st.markdown("### 📍 현재 대화 정보")
        slots = st.session_state.slots

        departure = slots.get("departure") or "미설정"
        arrival = slots.get("arrival") or "미설정"
        item = slots.get("item") or "미설정"

        st.markdown(f"""
                    <span class="slot-badge">🛫 출발: {departure}</span>
                    <span class="slot-badge">🛬 도착: {arrival}</span>
                    <span class="slot-badge">🎒 물품: {item}</span>
                    """, unsafe_allow_html=True)

        st.divider()

        if st.button("🔄 대화 초기화"):
            st.session_state.messages = []
            st.session_state.slots = {}
            st.rerun()

        st.divider()
        st.markdown("""
                    **사용 팁 💡**
                    - 노선을 먼저 알려주세요.
                    *예: 한국 → 미국 등*
                    - 물품명을 직접 입력하세요.
                    *예: 라이터, 보조배터리, 화장품 등*
                    - 노선이나 물품을 바꾸려면 새로운 노선과 물품을 입력해주세요.
                    """)
        
        st.markdown("---")

# 대화 히스토리 렌더링 함수 -> 지금까지의 대화 내용을 말풍선 형태로 출력
def render_chat_history() -> None:
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            render_user_message(message["content"])
        else:
            render_bot_message(message["content"])

# 로딩 메시지 렌더링 함수 -> LLM이 답변을 생성하는 동안 로딩 애니메이션과 안내 문구를 출력
def render_loading_message(placeholder) -> None:
    
    placeholder.markdown(LOADING_HTML, unsafe_allow_html=True)


# 봇 스트리밍 출력 
def stream_bot_response(placeholder, response_stream) -> str:
    
    full_response = ""

    for chunk in response_stream:
        full_response += chunk
        content_html = full_response.replace("\n", "<br>")

        placeholder.markdown(f"""
                            <div class="chat-bot">
                            <div class="sender-label">기내뭐돼 봇</div>
                            {content_html}
                            </div>
                            """, unsafe_allow_html=True)

    return full_response

# 사용자 입력 처리 
def handle_user_input(user_input: str) -> None:
   
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    render_user_message(user_input)

    placeholder = st.empty()
    render_loading_message(placeholder)

    response_stream, updated_slots = run_pipeline(
        user_message=user_input,
        chat_history=st.session_state.messages[:-1],
        slots=st.session_state.slots,
    )

    st.session_state.slots = updated_slots

    full_response = stream_bot_response(
        placeholder=placeholder,
        response_stream=response_stream,
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
    })

    st.rerun()

# 메인 함수
def main() -> None:
    st.set_page_config(
        page_title="기내뭐돼 ✈️",
        page_icon="✈️",
        layout="centered",
    )

    apply_custom_css()
    initialize_session_state()
    render_sidebar()
    render_header()
    render_welcome_message()
    render_chat_history()

    user_input = st.chat_input("노선과 물품을 입력하세요 (예: 한국→미국 고추장 반입 가능해?)")

    if user_input:
        handle_user_input(user_input)


if __name__ == "__main__":
    main()
