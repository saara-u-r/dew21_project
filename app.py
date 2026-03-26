import sys
import os
import json
import uuid
from datetime import datetime
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.rag import ask, ask_stream

# --- CHAT HISTORY PERSISTENCE ---
HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history")
os.makedirs(HISTORY_DIR, exist_ok=True)

def _history_path(chat_id: str) -> str:
    return os.path.join(HISTORY_DIR, f"{chat_id}.json")

def save_chat(chat_id: str, title: str, messages: list, sources: list):
    """Save a chat session to disk."""
    if not messages:
        return
    data = {
        "id": chat_id,
        "title": title,
        "updated_at": datetime.now().isoformat(),
        "messages": messages,
        "sources": sources,
    }
    with open(_history_path(chat_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_chat(chat_id: str) -> dict | None:
    """Load a single chat from disk."""
    path = _history_path(chat_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def delete_chat(chat_id: str):
    """Delete a chat file."""
    path = _history_path(chat_id)
    if os.path.exists(path):
        os.remove(path)

def list_chats() -> list[dict]:
    """List all saved chats, newest first."""
    chats = []
    for fname in os.listdir(HISTORY_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(HISTORY_DIR, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    chats.append({
                        "id": data["id"],
                        "title": data.get("title", "Untitled"),
                        "updated_at": data.get("updated_at", ""),
                    })
            except Exception:
                pass
    chats.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return chats

def generate_title(messages: list) -> str:
    """Generate a short title from the first user message."""
    for msg in messages:
        if msg["role"] == "user":
            text = msg["content"].strip()
            # Truncate to ~40 chars at word boundary
            if len(text) > 40:
                text = text[:37].rsplit(" ", 1)[0] + "..."
            return text
    return "New Chat"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="DEW21 Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME & UI TOKENS ---
UI = {
    "en": {
        "title": "DEW21 AI",
        "subtitle": "Energy Intelligence Platform",
        "description": "How can I help you today?",
        "placeholder": "Message DEW21 Assistant...",
        "clear": "New Chat",
        "thinking": "Searching documents...",
        "examples": ["Compare Electricity vs Gas rights", "What are the latest payment terms?", "Explain SCHUFA impact"],
        "lang_label": "🌐 EN"
    },
    "de": {
        "title": "DEW21 KI",
        "subtitle": "Energie-Intelligenz-Plattform",
        "description": "Wie kann ich dir heute helfen?",
        "placeholder": "DEW21 Assistent eine Nachricht senden...",
        "clear": "Neuer Chat",
        "thinking": "Dokumente werden durchsucht...",
        "examples": ["Strom vs. Gas Rechte vergleichen", "Zahlungsbedingungen prüfen", "SCHUFA-Auswirkungen erklären"],
        "lang_label": "🌐 DE"
    }
}

# --- SESSION STATE ---
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
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())
if "chat_title" not in st.session_state:
    st.session_state.chat_title = "New Chat"

t = UI[st.session_state.lang]

# --- CUSTOM CSS (THE MASTERPIECE) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700&display=swap');

/* Reset and Base */
html, body, [data-testid="stAppViewBlockContainer"] {
    background-color: #0d0d0d !important;
    color: #ececec !important;
    font-family: 'Inter', sans-serif !important;
}

/* Hide Streamlit elements */
header, footer, #MainMenu { visibility: hidden; }
.block-container { 
    padding: 0 !important; 
    max-width: 100% !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #000000 !important;
    border-right: 1px solid #222 !important;
    width: 260px !important;
}
[data-testid="stSidebarNav"] { display: none; }

.sidebar-header {
    padding: 20px 16px;
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.2rem;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sidebar-header span { color: #f57c00; }

.sidebar-section-title {
    padding: 24px 16px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.history-item {
    padding: 10px 16px;
    margin: 2px 8px;
    border-radius: 8px;
    font-size: 0.9rem;
    color: #ccc;
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: all 0.2s;
}
.history-item:hover {
    background: #1a1a1a;
    color: #fff;
}

/* Main Layout */
.main-chat-area {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    min-height: 100vh;
}

.chat-container {
    width: 100%;
    max-width: 800px;
    padding: 80px 20px 160px;
}

/* Model Selector (Modern ChatGPT Style) */
div[data-testid="stPopover"] {
    position: fixed !important;
    bottom: 90px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    z-index: 1000 !important;
    width: auto !important;
}

div[data-testid="stPopover"] button {
    background: rgba(20, 20, 20, 0.7) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 8px 16px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    color: #fff !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
}
div[data-testid="stPopover"] button:hover {
    background: rgba(40, 40, 40, 0.8) !important;
    border-color: rgba(245, 124, 0, 0.5) !important;
}

/* Hero Section */
.hero {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 60vh;
    text-align: center;
}
.hero-logo {
    font-family: 'Outfit', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 24px;
    color: #fff;
}
.hero-logo span { color: #f57c00; }

.example-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    max-width: 600px;
    width: 100%;
}

/* Message Styling */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: 24px 0 !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
}
.stChatMessage[data-testid="chat-message-user"] {
    background: rgba(255, 255, 255, 0.02) !important;
}

[data-testid="stMarkdownContainer"] p {
    font-size: 1rem !important;
    line-height: 1.7 !important;
    color: #d1d1d1 !important;
}

/* Chat Input Area */
.stChatInputContainer {
    background: transparent !important;
    border: none !important;
    padding-bottom: 40px !important;
}
.stChatInputContainer > div {
    background: #1e1e1e !important;
    border: 1px solid #333 !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5) !important;
    max-width: 800px !important;
    margin: 0 auto !important;
    transition: border-color 0.3s;
}
.stChatInputContainer:focus-within > div {
    border-color: #f57c00 !important;
}

/* Custom Searching Animation */
@keyframes shimmer {
    0% { background-position: -468px 0; }
    100% { background-position: 468px 0; }
}
.searching-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: linear-gradient(to right, rgba(245, 124, 0, 0.05) 8%, rgba(245, 124, 0, 0.2) 18%, rgba(245, 124, 0, 0.05) 33%);
    background-size: 800px 104px;
    animation: shimmer 2s infinite linear;
    border: 1px solid rgba(245, 124, 0, 0.3);
    border-radius: 100px;
    color: #f57c00;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 16px;
}

/* Expander (Citations) */
.streamlit-expanderHeader {
    background: transparent !important;
    color: #666 !important;
    font-size: 0.8rem !important;
    border: none !important;
}
.streamlit-expanderHeader:hover {
    color: #f57c00 !important;
}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(f'<div class="sidebar-header">DEW<span>21</span> {t["title"]}</div>', unsafe_allow_html=True)
    
    # New Chat button — saves current chat first
    if st.button(f"➕ {t['clear']}", use_container_width=True, type="secondary"):
        # Save current conversation before starting new one
        if st.session_state.messages:
            title = generate_title(st.session_state.messages)
            save_chat(st.session_state.chat_id, title, st.session_state.messages, st.session_state.sources)
        # Reset to a fresh chat
        st.session_state.messages = []
        st.session_state.sources = []
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.chat_title = "New Chat"
        st.rerun()
        
    st.markdown('<div class="sidebar-section-title">History</div>', unsafe_allow_html=True)
    
    # Real persistent history
    saved_chats = list_chats()
    for chat_meta in saved_chats:
        cid = chat_meta["id"]
        is_active = cid == st.session_state.chat_id
        
        col_title, col_del = st.columns([5, 1])
        with col_title:
            label = f"💬 {chat_meta['title']}" if not is_active else f"▶ {chat_meta['title']}"
            if st.button(label, key=f"load_{cid}", use_container_width=True):
                # Save current chat before switching
                if st.session_state.messages:
                    save_chat(st.session_state.chat_id, generate_title(st.session_state.messages), st.session_state.messages, st.session_state.sources)
                # Load the selected chat
                chat_data = load_chat(cid)
                if chat_data:
                    st.session_state.chat_id = chat_data["id"]
                    st.session_state.chat_title = chat_data.get("title", "Untitled")
                    st.session_state.messages = chat_data.get("messages", [])
                    st.session_state.sources = chat_data.get("sources", [])
                    st.rerun()
        with col_del:
            if st.button("🗑", key=f"del_{cid}"):
                delete_chat(cid)
                # If we deleted the active chat, reset
                if is_active:
                    st.session_state.messages = []
                    st.session_state.sources = []
                    st.session_state.chat_id = str(uuid.uuid4())
                    st.session_state.chat_title = "New Chat"
                st.rerun()
    
    if not saved_chats:
        st.caption("No saved chats yet.")
        
    st.markdown('<div style="margin-top: auto;"></div>', unsafe_allow_html=True)
    
    # Bottom Sidebar Settings
    with st.expander("🌐 Language & Region"):
        if st.button("English 🇺🇸", use_container_width=True):
            st.session_state.lang = "en"
            st.rerun()
        if st.button("Deutsch 🇩🇪", use_container_width=True):
            st.session_state.lang = "de"
            st.rerun()



# --- MAIN CHAT AREA ---
st.markdown('<div class="main-chat-area">', unsafe_allow_html=True)

# Container for history and hero
chat_content = st.container()

# Wrap history in centered container
st.markdown('<div style="max-width: 800px; margin: 0 auto; width: 100%;">', unsafe_allow_html=True)

with chat_content:
    if not st.session_state.messages:
        # Hero Section
        st.markdown(f'''
        <div class="hero">
            <div class="hero-logo">DEW<span>21</span></div>
            <p style="font-size: 1.5rem; font-weight: 500; color: #fff; margin-bottom: 32px;">{t["description"]}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Example Buttons
        ex_cols = st.columns([1, 1, 1])
        for i, ex in enumerate(t["examples"]):
            if ex_cols[i % 3].button(ex, use_container_width=True):
                st.session_state.example_trigger = ex
                st.rerun()
    else:
        # Chat Messages
        for i, msg in enumerate(st.session_state.messages):
            role = msg["role"]
            avatar = "⚡" if role == "assistant" else None
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg["content"])
                
                if role == "assistant" and i < len(st.session_state.sources):
                    sdata = st.session_state.sources[i]
                    if sdata and sdata.get("highlights"):
                        with st.expander("View Citations"):
                            for h in sdata["highlights"]:
                                st.caption(f'"{h["text"]}"')
                                st.markdown(f"<small style='color:#555'>— {h['source']}</small>", unsafe_allow_html=True)

# --- CHAT INPUT ---
user_query = st.chat_input(t["placeholder"], disabled=st.session_state.generating)

if hasattr(st.session_state, "example_trigger") and st.session_state.example_trigger:
    user_query = st.session_state.example_trigger
    del st.session_state.example_trigger

if user_query:
    st.session_state.generating = True
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.sources.append(None)
    st.rerun()

# --- GENERATION LOGIC ---
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and st.session_state.generating:
    last_query = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant", avatar="⚡"):
        with st.empty():
            st.markdown('<div class="searching-badge"><span>🔍</span> Searching documents...</div>', unsafe_allow_html=True)
            
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
            snippet = d.page_content.replace('\n', ' ')
            if len(snippet) > 150:
                snippet = snippet[:147] + "..."
            highlights.append({"text": snippet, "source": source})

    st.session_state.messages.append({"role": "assistant", "content": assistant_content})
    st.session_state.sources.append({"highlights": highlights})
    st.session_state.generating = False
    
    # Auto-save after every assistant response
    title = generate_title(st.session_state.messages)
    st.session_state.chat_title = title
    save_chat(st.session_state.chat_id, title, st.session_state.messages, st.session_state.sources)
    
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True) # Close centered container
st.markdown('</div>', unsafe_allow_html=True) # Close main-chat-area

