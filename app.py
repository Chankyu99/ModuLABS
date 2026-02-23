"""
app.py
------
'기내뭐돼' Streamlit 챗봇 UI

실행:
    .venv/bin/streamlit run app.py
"""

import streamlit as st
from bot_logic import run_pipeline

# ── 페이지 기본 설정 ────────────────────────────────────────────
st.set_page_config(
    page_title="기내뭐돼 ✈️",
    page_icon="✈️",
    layout="centered",
)

# ── 커스텀 CSS ──────────────────────────────────────────────────
st.markdown("""
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
""", unsafe_allow_html=True)

# ── 세션 상태 초기화 ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "slots" not in st.session_state:
    st.session_state.slots = {}

# ── 사이드바 ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ 기내뭐돼")
    st.markdown("항공 반입 규정 RAG 챗봇")
    st.divider()

    # 현재 슬롯 상태 표시
    st.markdown("### 📍 현재 대화 정보")
    slots = st.session_state.slots

    dep = slots.get("departure") or "미설정"
    arr = slots.get("arrival")   or "미설정"
    itm = slots.get("item")      or "미설정"

    st.markdown(f"""
<span class="slot-badge">🛫 출발: {dep}</span>
<span class="slot-badge">🛬 도착: {arr}</span>
<span class="slot-badge">🎒 물품: {itm}</span>
""", unsafe_allow_html=True)

    st.divider()

    if st.button("🔄 대화 초기화"):
        st.session_state.messages = []
        st.session_state.slots    = {}
        st.rerun()

    st.divider()
    st.markdown("""
**사용 팁 💡**
- 노선을 먼저 알려주세요
  *예: 한국 → 미국*
- 물품명을 직접 입력하세요
  *예: 고추장, 보조배터리*
- 노선을 바꾸려면 새로 말해주세요
""")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#555;'>Powered by LangChain + ChromaDB + GPT-4o-mini</div>",
        unsafe_allow_html=True,
    )

# ── 메인 헤더 ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">✈️ 기내뭐돼</div>
  <p class="hero-subtitle">공식 항공 규정 기반 · AI 반입 가능 여부 안내 서비스</p>
</div>
""", unsafe_allow_html=True)

# ── 초기 안내 메시지 ──────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
<div class="chat-bot">
  <div class="sender-label">기내뭐돼 봇</div>
  안녕하세요! 저는 항공 반입 규정 안내 챗봇 <b>기내뭐돼</b>입니다. 🛫<br><br>
  궁금하신 <b>노선</b>과 <b>물품</b>을 알려주시면 기내/위탁 반입 가능 여부를 안내해 드릴게요.<br><br>
  예시 질문:<br>
  • <i>"한국에서 미국 갈 때 고추장 가져갈 수 있어?"</i>
</div>
""", unsafe_allow_html=True)

# ── 대화 히스토리 렌더링 ──────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
<div class="chat-user">
  <div class="sender-label" style="color:rgba(255,255,255,0.6);">나</div>
  {msg["content"]}
</div>
""", unsafe_allow_html=True)
    else:
        # 줄바꿈 처리
        content = msg["content"].replace("\n", "<br>")
        st.markdown(f"""
<div class="chat-bot">
  <div class="sender-label">기내뭐돼 봇</div>
  {content}
</div>
""", unsafe_allow_html=True)

# ── 채팅 입력 ─────────────────────────────────────────────────────
if user_input := st.chat_input("노선과 물품을 입력하세요 (예: 한국→미국 고추장 반입 가능해?)"):

    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"""
<div class="chat-user">
  <div class="sender-label" style="color:rgba(255,255,255,0.6);">나</div>
  {user_input}
</div>
""", unsafe_allow_html=True)

    # 로딩 스피너 (슬롯 추출 및 DB 검색까지 대기)
    with st.spinner("🔍 항공 규정 검색 중..."):
        bot_response_stream, updated_slots = run_pipeline(
            user_message=user_input,
            chat_history=st.session_state.messages[:-1],  # 방금 추가한 것 제외
            slots=st.session_state.slots,
        )

    # 슬롯 상태 업데이트
    st.session_state.slots = updated_slots

    # 스트리밍 응답 렌더링
    placeholder = st.empty()
    full_response = ""
    
    # 첫 토큰이 오기 전까지 대기하는 동안 보여줄 임시 메시지
    placeholder.markdown("""
<div class="chat-bot">
  <div class="sender-label">기내뭐돼 봇</div>
  🤔 규정 검토 및 답변 작성 중...
</div>
""", unsafe_allow_html=True)

    first_chunk = True
    for chunk in bot_response_stream:
        if first_chunk:
            full_response = ""
            first_chunk = False
            
        full_response += chunk
        content_html = full_response.replace("\n", "<br>")
        placeholder.markdown(f"""
<div class="chat-bot">
  <div class="sender-label">기내뭐돼 봇</div>
  {content_html}
</div>
""", unsafe_allow_html=True)

    # 스트리밍 완료 후 메시지 히스토리에 최종 저장
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()
