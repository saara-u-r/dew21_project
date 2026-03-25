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
if "mode" not in st.session_state:
    st.session_state.mode = "Standard"
if "context" not in st.session_state:
    st.session_state.context = "All Docs"
if "generating" not in st.session_state:
    st.session_state.generating = False

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

/* ===== POPOVER (Language Dropdown) ===== */
div[data-testid="stPopover"] button {
    background: #161616 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #c0c0c0 !important;
    font-size: 13px !important;
    padding: 6px 10px !important;
    width: 100% !important;
}
div[data-testid="stPopover"] button:hover {
    border-color: #F57C00 !important;
    color: #F57C00 !important;
}
div[data-testid="stPopoverContent"] button {
    text-align: left !important;
    justify-content: flex-start !important;
    border: none !important;
    background: transparent !important;
    padding: 8px 12px !important;
    color: #fff !important;
}
div[data-testid="stPopoverContent"] button:hover {
    background: #1f1400 !important;
    color: #F57C00 !important;
}
/* ===== HIDE STREAMLIT CHROME & DIMMING ===== */
div[data-testid="stAppViewBlockContainer"] > div:last-child {
    background: transparent !important;
}
[data-testid="stAppViewBlockContainer"] {
    opacity: 1 !important; /* Forces full opacity even during run */
}
.stActionButton { display: none !important; }

/* Custom Pulse for Searching */
@keyframes pulse {
    0% { opacity: 0.4; }
    50% { opacity: 1; }
    100% { opacity: 0.4; }
}
.searching-pulse {
    animation: pulse 1.5s infinite ease-in-out;
    color: #F57C00;
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 20px;
}

/* THE UNIFIED COMMAND BAR */
.controls-container {
    position: fixed !important;
    bottom: 90px !important;
    left: 20px !important; /* Left-aligned like ChatGPT tools */
    z-index: 1000000 !important;
}

div[data-testid="stPopover"] button {
    background: rgba(15, 15, 15, 0.95) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0 16px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.6) !important;
    height: 38px !important;
}

div[data-testid="stPopover"] button:hover {
    background: rgba(40, 40, 40, 0.9) !important;
    border-color: #F57C00 !important;
    color: #F57C00 !important;
    box-shadow: 0 0 15px rgba(245, 124, 0, 0.1) !important;
}

/* Fix for the popover content to match theme */
div[data-testid="stPopoverContent"] {
    background: #141414 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.8) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Top Nav
col_logo, col_spacer, col_lang, col_clear = st.columns([3, 2, 1, 1.4])
with col_logo:
    st.markdown('<div class="nav-logo">DEW<span>21</span></div>', unsafe_allow_html=True)
with col_lang:
    with st.popover(f"{st.session_state.lang.upper()} 🌐", use_container_width=True):
        if st.button("English 🇺🇸", use_container_width=True):
            st.session_state.lang = "en"
            st.rerun()
        if st.button("Deutsch 🇩🇪", use_container_width=True):
            st.session_state.lang = "de"
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

user_query = st.chat_input(t["placeholder"], disabled=st.session_state.generating)

# --- UNIFIED SETTINGS (MODE + CONTEXT) ---
st.markdown('<div class="controls-container">', unsafe_allow_html=True)
mode_icons = {"Simplified": "📄", "Standard": "⚖️", "Expert": "🎓"}
ctx_icons = {"All Docs": "📚", "Electricity": "⚡", "Gas": "🔥", "SCHUFA": "🏦", "Creditreform": "💳"}

with st.popover(f"⚙️ {st.session_state.mode} • {st.session_state.context}", use_container_width=False):
    st.write("**Target Mode**")
    for m in ["Simplified", "Standard", "Expert"]:
        if st.button(f"{mode_icons[m]} {m}", key=f"btn_mode_{m}", use_container_width=True):
            st.session_state.mode = m
            st.rerun()
    st.write("---")
    st.write("**Document Context**")
    for c in ["All Docs", "Electricity", "Gas", "SCHUFA", "Creditreform"]:
        if st.button(f"{ctx_icons[c]} {c}", key=f"btn_ctx_{c}", use_container_width=True):
            st.session_state.context = c
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

if hasattr(st.session_state, "example_trigger") and st.session_state.example_trigger:
    user_query = st.session_state.example_trigger
    del st.session_state.example_trigger

if user_query:
    # ── Immediate state update
    st.session_state.generating = True
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.sources.append(None)
    st.rerun()

# ── Processing the last message if needed
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and st.session_state.generating:
    last_query = st.session_state.messages[-1]["content"]
    
    with chat_container.chat_message("assistant", avatar="⚡"):
        # Use st.write_stream for real-time feedback
        from src.rag import ask_stream
        
        # Display a small badge while initializing
        with st.empty():
            st.markdown('<div class="searching-pulse">🔍 Searching documents...</div>', unsafe_allow_html=True)
            
            # Stream the answer
            retrieved_docs = []
            assistant_content = st.write_stream(
                ask_stream(
                    last_query, 
                    chat_history=st.session_state.messages[:-1], 
                    lang=st.session_state.lang, 
                    retrieved_docs_out=retrieved_docs,
                    mode=st.session_state.mode,
                    doc_filter=st.session_state.context
                )
            )
            
        highlights = []
        for d in retrieved_docs:
            source = d.metadata.get("doc_name", "Unknown Document")
            # Create a short snippet
            snippet = d.page_content.replace('\n', ' ')
            if len(snippet) > 150:
                snippet = snippet[:147] + "..."
            highlights.append({"text": snippet, "source": source})

    st.session_state.messages.append({"role": "assistant", "content": assistant_content})
    st.session_state.sources.append({"highlights": highlights})
    st.session_state.generating = False
    st.rerun()
