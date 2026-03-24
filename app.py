import sys
import os
import streamlit as st

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag import ask

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="DEW21 GenAI Assistant",
    page_icon="⚡",
    layout="wide",  # 🔥 FULL SCREEN MODE
    initial_sidebar_state="expanded"
)

# --- TRANSLATIONS ---
UI = {
    "en": {
        "title": "DEW21 GenAI Assistant",
        "description": "Ask about your energy contracts, billing, and rights.",
        "placeholder": "Ask a question about DEW21 contracts...",
        "clear": "🗑️ Clear Chat",
        "thinking": "Searching documents...",
        "evidence": "🔍 Supporting Evidence",
        "badge": "AI-ASSISTANT",
        "examples": ["What is SCHUFA?", "How can I terminate my contract?", "What are the payment terms?"]
    },
    "de": {
        "title": "DEW21 KI-Assistent",
        "description": "Fragen zu Energielieferverträgen, Abrechnungen und Rechten.",
        "placeholder": "Stellen Sie Ihre Frage zu DEW21-Verträgen...",
        "clear": "🗑️ Verlauf löschen",
        "thinking": "Suche in Dokumenten...",
        "evidence": "🔍 Belege aus den Dokumenten",
        "badge": "KI-ASSISTENT",
        "examples": ["Was ist SCHUFA?", "Wie kann ich meinen Vertrag kündigen?", "Was sind die Zahlungsbedingungen?"]
    }
}

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Chat Message Styling */
    .stChatMessage {
        background-color: #ffffff !important;
        border: 1px solid #e1e4e8 !important;
        padding: 10px 20px !important;
        border-radius: 12px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
    }
    
    /* Ensure ALL text elements in chat are readable with high contrast black */
    .stChatMessage [data-testid="stMarkdownContainer"] p,
    .stChatMessage [data-testid="stMarkdownContainer"] li,
    .stChatMessage [data-testid="stMarkdownContainer"] span {
        color: #000000 !important;
        font-weight: 500 !important;
        line-height: 1.6 !important;
    }
    
    .stChatMessage a {
        color: #004494 !important;
        font-weight: bold !important;
        text-decoration: underline !important;
    }

    /* DEW21 Header */
    .dew-header {
        background: #ffffff;
        padding: 15px 30px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 4px solid #F57C00;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .dew-logo { 
        font-size: 32px; 
        font-weight: 900; 
        color: #002F6C; 
        letter-spacing: -1.5px;
        text-transform: uppercase;
    }
    .dew-logo span {
        color: #E30613;
    }
    .dew-logo-sub {
        font-weight: 500;
        color: #002F6C;
        font-size: 12px;
        opacity: 0.8;
    }
    
    /* Hero Banner - Fit inside container cleanly */
    .dew-hero {
        background: linear-gradient(135deg, #001E44 0%, #002F6C 100%);
        padding: 30px;
        color: white;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .hero-badge {
        background: #F57C00;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        display: inline-block;
        font-weight: bold;
        margin-bottom: 8px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #001E44 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = []
if "lang" not in st.session_state:
    st.session_state.lang = "en"

t = UI[st.session_state.lang]

# --- UI ELEMENTS ---

# Header
st.markdown(f'''
<div class="dew-header">
    <div class="dew-logo">
        DEW<span style="color:#F57C00;">21</span>
    </div>
    <div class="dew-logo-sub">STROM | GAS | WÄRME | WASSER</div>
</div>
<div style="margin-top: 20px;"></div>
''', unsafe_allow_html=True)

# Hero (Full Width)
st.markdown(f'<div class="dew-hero"><div class="hero-badge">{t["badge"]}</div><h1 style="color:white; margin:0;">{t["title"]}</h1><p style="color:rgba(255,255,255,0.8);">{t["description"]}</p></div>', unsafe_allow_html=True)

# --- CHAT DISPLAY ---
chat_container = st.container()
with chat_container:
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "⚡"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and i < len(st.session_state.sources):
                sdata = st.session_state.sources[i]
                if sdata and sdata.get("highlights"):
                    with st.expander(t["evidence"]):
                        for h in sdata["highlights"]:
                            st.caption(f"📁 Source: {h['source']}")
                            st.write(f"_{h['text']}_")

# --- CHAT INPUT ---
if query := st.chat_input(t["placeholder"]):
    with chat_container.chat_message("user", avatar="🧑‍💼"):
        st.markdown(query)
    
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.sources.append(None)
    
    with chat_container.chat_message("assistant", avatar="⚡"):
        with st.spinner(t["thinking"]):
            res = ask(query, chat_history=st.session_state.messages[:-1], lang=st.session_state.lang)
            st.markdown(res["answer"])
            if res.get("highlights"):
                with st.expander(t["evidence"]):
                    for h in res["highlights"]:
                        st.caption(f"📁 Source: {h['source']}")
                        st.write(f"_{h['text']}_")
            
            st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
            st.session_state.sources.append({"highlights": res.get("highlights", [])})

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<h2 style="color:white;">DEW<span style="color:#F57C00;">21</span></h2>', unsafe_allow_html=True)
    st.subheader("🌐 Language / Sprache")
    new_lang = st.selectbox("Select Language", options=["en", "de"], format_func=lambda x: "English 🇺🇸" if x=="en" else "Deutsch 🇩🇪")
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()
    
    st.divider()
    st.subheader("💡 Examples")
    for ex in t["examples"]:
        if st.button(ex, key=ex):
            # Programmatically trigger chat if needed
            pass
            
    if st.button(t["clear"], use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources = []
        st.rerun()
