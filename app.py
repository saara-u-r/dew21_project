import sys
import os
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.rag import ask

st.set_page_config(
    page_title="DEW21 Assistant",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

UI = {
    "en": {
        "title": "DEW21 <span>⚡</span> Assistant",
        "description": "Ask about your energy contracts, billing, and legal rights.",
        "placeholder": "Ask DEW21 Assistant...",
        "clear": "New Chat",
        "thinking": "Searching documents...",
        "examples": ["What is SCHUFA?", "Can I terminate my contract?", "What are the payment terms?"],
        "lang_label": "🌐 EN"
    },
    "de": {
        "title": "DEW21 <span>⚡</span> Assistent",
        "description": "Fragen zu Energieverträgen, Abrechnung und Ihren Rechten.",
        "placeholder": "DEW21 Assistent fragen...",
        "clear": "Neuer Chat",
        "thinking": "Dokumente werden durchsucht...",
        "examples": ["Was ist die SCHUFA?", "Kann ich meinen Vertrag kündigen?", "Was sind die Zahlungsbedingungen?"],
        "lang_label": "🌐 DE"
    }
}

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = []
if "lang" not in st.session_state:
    st.session_state.lang = "en"

t = UI[st.session_state.lang]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ===== BASE ===== */
html, body, .stApp {
    background-color: #0d0d0d !important;
    color: #f5f5f5 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Hide all default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; height: 0 !important; }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 120px !important;
    max-width: 780px !important;
}

/* ===== TOP NAV BAR ===== */
.top-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 0 14px;
    border-bottom: 1px solid #1f1f1f;
    margin-bottom: 36px;
}
.nav-logo {
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #ffffff;
}
.nav-logo span { color: #F57C00; }

/* ===== HERO AREA ===== */
.hero {
    text-align: center;
    padding: 20px 0 36px;
}
.hero h1 {
    font-size: 40px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #ffffff;
    margin-bottom: 8px;
    line-height: 1.1;
}
.hero h1 span { color: #F57C00; }
.hero p {
    font-size: 15px;
    color: #888;
    font-weight: 400;
    max-width: 420px;
    margin: 0 auto;
}

/* ===== CHAT MESSAGES ===== */
.stChatMessage {
    background: #161616 !important;
    border: 1px solid #252525 !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    margin-bottom: 14px !important;
    box-shadow: none !important;
}
.stChatMessage [data-testid="stMarkdownContainer"] p {
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.65 !important;
    color: #e8e8e8 !important;
}

/* User bubble — orange tint */
.stChatMessage[data-testid="chat-message-user"] {
    background: #1a1200 !important;
    border: 1px solid #3d2800 !important;
    border-radius: 14px 14px 4px 14px !important;
    margin-left: auto;
    width: fit-content;
    max-width: 85%;
}
.stChatMessage[data-testid="chat-message-user"] [data-testid="stMarkdownContainer"] p {
    color: #ffd180 !important;
}

/* Assistant bubble — subtle left accent */
.stChatMessage[data-testid="chat-message-assistant"] {
    border-left: 3px solid #F57C00 !important;
    border-radius: 4px 14px 14px 14px !important;
}

/* ===== EXPANDER (Citations) ===== */
.streamlit-expanderHeader {
    font-size: 13px !important;
    color: #F57C00 !important;
    background: transparent !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
}
.streamlit-expanderContent {
    background: #0d0d0d !important;
    border: 1px solid #222 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    font-size: 13px !important;
    color: #aaa !important;
}

/* ===== EXAMPLE BUTTONS ===== */
div[data-testid="stButton"] button {
    background: #161616 !important;
    border: 1px solid #2a2a2a !important;
    color: #c0c0c0 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stButton"] button:hover {
    background: #1f1400 !important;
    border-color: #F57C00 !important;
    color: #F57C00 !important;
}

/* ===== CHAT INPUT BAR ===== */
.stChatInputContainer {
    background: #161616 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 14px !important;
    padding: 6px !important;
    max-width: 780px !important;
    margin: 0 auto !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.6) !important;
    bottom: 20px !important;
}
.stChatInputContainer:focus-within {
    border-color: #F57C00 !important;
}
.stChatInputContainer textarea {
    background: transparent !important;
    font-size: 15px !important;
    color: #f0f0f0 !important;
    padding: 8px 14px !important;
    line-height: 1.5 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Send button glow */
.stChatInputContainer button {
    color: #F57C00 !important;
}

/* ===== SPINNER ===== */
.stSpinner p { color: #888 !important; font-size: 14px !important; }

/* ===== SELECTBOX (Language) ===== */
div[data-testid="stSelectbox"] > div {
    background: #161616 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #c0c0c0 !important;
    font-size: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Top Nav
col_logo, col_spacer, col_lang, col_clear = st.columns([3, 2, 1, 1.4])
with col_logo:
    st.markdown('<div class="nav-logo">DEW<span>21</span></div>', unsafe_allow_html=True)
with col_lang:
    new_lang = st.selectbox(
        "lang",
        options=["en", "de"],
        index=0 if st.session_state.lang == "en" else 1,
        format_func=lambda x: "EN 🌐" if x == "en" else "DE 🌐",
        label_visibility="collapsed"
    )
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        t = UI[new_lang]
        st.rerun()
with col_clear:
    if st.button(t["clear"], use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources = []
        st.rerun()

# ── Hero Section (only when chat is empty)
if not st.session_state.messages:
    st.markdown(f'''
    <div class="hero">
        <h1>{t["title"]}</h1>
        <p>{t["description"]}</p>
    </div>
    ''', unsafe_allow_html=True)

    # ── Example Chips
    cols = st.columns(3)
    for i, ex in enumerate(t["examples"]):
        if cols[i].button(ex, use_container_width=True):
            st.session_state.example_trigger = ex
            st.rerun()

# ── Chat history
chat_container = st.container()
with chat_container:
    for i, msg in enumerate(st.session_state.messages):
        avatar = "🧑" if msg["role"] == "user" else "⚡"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and i < len(st.session_state.sources):
                sdata = st.session_state.sources[i]
                if sdata and sdata.get("highlights"):
                    with st.expander("📎 View Citations"):
                        for h in sdata["highlights"]:
                            st.caption(f'"{h["text"]}"')
                            st.markdown(f"<small style='color:#555'>— {h['source']}</small>", unsafe_allow_html=True)

# ── Input
user_query = st.chat_input(t["placeholder"])

if hasattr(st.session_state, "example_trigger") and st.session_state.example_trigger:
    user_query = st.session_state.example_trigger
    del st.session_state.example_trigger

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.sources.append(None)

    with chat_container.chat_message("user", avatar="🧑"):
        st.markdown(user_query)

    with chat_container.chat_message("assistant", avatar="⚡"):
        with st.spinner(t["thinking"]):
            res = ask(user_query, chat_history=st.session_state.messages[:-1], lang=st.session_state.lang)
            st.markdown(res["answer"])
            highlights = res.get("highlights", [])
            if highlights:
                with st.expander("📎 View Citations"):
                    for h in highlights:
                        st.caption(f'"{h["text"]}"')
                        st.markdown(f"<small style='color:#555'>— {h['source']}</small>", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
    st.session_state.sources.append({"highlights": highlights})
