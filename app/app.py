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
    path = _history_path(chat_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def delete_chat(chat_id: str):
    path = _history_path(chat_id)
    if os.path.exists(path):
        os.remove(path)

def list_chats() -> list[dict]:
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
    for msg in messages:
        if msg["role"] == "user":
            text = msg["content"].strip()
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

# --- UI TOKENS ---
UI = {
    "en": {
        "subtitle": "Smarter answers for energy contracts, billing, and policies.",
        "placeholder": "Ask about contracts, billing, SCHUFA, or policies...",
        "clear": "New Chat",
        "examples": [
            {"icon": "⚡", "title": "Compare Electricity vs Gas", "sub": "Pricing, usage rights & key differences"},
            {"icon": "📄", "title": "Latest payment terms", "sub": "Due dates, penalties & grace periods"},
            {"icon": "🏦", "title": "SCHUFA impact explained", "sub": "Credit checks and what they mean"},
            {"icon": "📋", "title": "Contract cancellation rules", "sub": "Notice periods and legal obligations"},
        ],
        "modes": [
            {"key": "Simplified", "icon": "📄", "desc": "Plain language, easy to understand"},
            {"key": "Standard",   "icon": "⚖️", "desc": "Balanced detail and clarity"},
            {"key": "Expert",     "icon": "🎓", "desc": "Full technical and legal depth"},
        ],
        "followups": [
            "What are the penalty clauses?",
            "How does this affect billing?",
            "Are there exceptions to this rule?",
        ],
    },
    "de": {
        "subtitle": "Smarte Antworten zu Energieverträgen, Abrechnung und Richtlinien.",
        "placeholder": "Fragen zu Verträgen, Abrechnung, SCHUFA oder Richtlinien...",
        "clear": "Neuer Chat",
        "examples": [
            {"icon": "⚡", "title": "Strom vs. Gas vergleichen", "sub": "Preise, Nutzungsrechte & Unterschiede"},
            {"icon": "📄", "title": "Aktuelle Zahlungsbedingungen", "sub": "Fälligkeiten, Strafen & Fristen"},
            {"icon": "🏦", "title": "SCHUFA-Auswirkungen", "sub": "Bonitätsprüfungen erklärt"},
            {"icon": "📋", "title": "Vertragskündigung", "sub": "Kündigungsfristen & Pflichten"},
        ],
        "modes": [
            {"key": "Simplified", "icon": "📄", "desc": "Einfache Sprache, leicht verständlich"},
            {"key": "Standard",   "icon": "⚖️", "desc": "Ausgewogene Details und Klarheit"},
            {"key": "Expert",     "icon": "🎓", "desc": "Volle technische und rechtliche Tiefe"},
        ],
        "followups": [
            "Welche Strafklauseln gibt es?",
            "Wie beeinflusst das die Abrechnung?",
            "Gibt es Ausnahmen?",
        ],
    },
}

# --- SESSION STATE ---
defaults = {
    "messages": [],
    "sources": [],
    "lang": "en",
    "mode": "Standard",
    "context": "All Docs",
    "generating": False,
    "chat_id": str(uuid.uuid4()),
    "chat_title": "New Chat",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

from typing import Any
t: Any = UI[st.session_state.lang]

# Sync mode if language switched
mode_keys = [m["key"] for m in t["modes"]]  # type: ignore
if st.session_state.mode not in mode_keys:
    st.session_state.mode = mode_keys[0]

# =========================================================
# MASTER CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewBlockContainer"] {
    background: linear-gradient(160deg, #0B0F14 0%, #0E141B 100%) !important;
    color: #E5E7EB !important;
    font-family: 'Inter', sans-serif !important;
}

header, footer, #MainMenu { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #080C10 !important;
    border-right: 1px solid #1F2937 !important;
    width: 272px !important;
}
[data-testid="stSidebarNav"] { display: none; }

.sb-logo {
    padding: 24px 18px 16px;
    display: flex;
    align-items: flex-start;
    gap: 8px;
}
.sb-logo-wordmark {
    font-family: 'Outfit', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.02em;
    line-height: 1;
}
.sb-logo-wordmark em { color: #FF6A00; font-style: normal; }
.sb-logo-badge {
    font-size: 0.58rem;
    font-weight: 700;
    color: #FF6A00;
    background: rgba(255,106,0,0.1);
    border: 1px solid rgba(255,106,0,0.3);
    border-radius: 4px;
    padding: 2px 5px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 1px;
}
.sb-status {
    margin: 0 14px 14px;
    padding: 8px 12px;
    background: rgba(16,185,129,0.06);
    border: 1px solid rgba(16,185,129,0.18);
    border-radius: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.73rem;
    color: #6EE7B7;
    font-weight: 500;
}
.sb-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 6px #10B981;
    flex-shrink: 0;
}
.sb-section {
    padding: 18px 18px 6px;
    font-size: 0.67rem;
    font-weight: 700;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="stSidebar"] button {
    background: transparent !important;
    border: none !important;
    color: #6B7280 !important;
    text-align: left !important;
    font-size: 0.84rem !important;
    border-radius: 8px !important;
    transition: all 0.15s !important;
}
[data-testid="stSidebar"] button:hover {
    background: #111827 !important;
    color: #F9FAFB !important;
}

/* ── Main layout ── */
.main-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
}

/* ── Hero ── */
.hero-wrap {
    max-width: 720px;
    width: 100%;
    margin: 0 auto;
    padding: 64px 24px 28px;
}
.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    color: #FF6A00;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 12px;
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: #F9FAFB;
    line-height: 1.15;
    letter-spacing: -0.03em;
    margin-bottom: 14px;
}
.hero-title em { color: #FF6A00; font-style: normal; }
.hero-sub {
    font-size: 1rem;
    color: #6B7280;
    line-height: 1.65;
    max-width: 500px;
    margin-bottom: 36px;
}

/* Card buttons — hide Streamlit default styling, show our card on top */
.cards-outer {
    max-width: 720px;
    margin: 0 auto;
    padding: 0 24px 32px;
}
/* We show a visible HTML card above and a near-invisible Streamlit button below */
.scard {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 12px;
    padding: 15px 16px;
    margin-bottom: -6px;   /* pulled tight so button sits over card */
    pointer-events: none;  /* button handles the click */
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.scard-icon { font-size: 1.25rem; flex-shrink: 0; margin-top: 1px; }
.scard-title { font-size: 0.88rem; font-weight: 600; color: #F3F4F6; margin-bottom: 3px; }
.scard-sub   { font-size: 0.73rem; color: #6B7280; line-height: 1.4; }

/* ── Chat messages ── */
.chat-wrap {
    max-width: 760px;
    width: 100%;
    margin: 0 auto;
    padding: 36px 24px 200px;
}
.stChatMessage[data-testid="chat-message-user"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid #1F2937 !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin-bottom: 6px !important;
}
.stChatMessage[data-testid="chat-message-assistant"] {
    background: transparent !important;
    border: none !important;
    padding: 6px 0 20px !important;
    border-bottom: 1px solid rgba(255,255,255,0.04) !important;
    margin-bottom: 6px !important;
}
[data-testid="stMarkdownContainer"] p {
    font-size: 0.96rem !important;
    line-height: 1.78 !important;
    color: #D1D5DB !important;
}

/* ── Source cards ── */
.source-card {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    background: #0F1623;
    border: 1px solid #1F2937;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.src-icon  { font-size: 1rem; flex-shrink: 0; margin-top: 2px; }
.src-name  { font-size: 0.76rem; font-weight: 600; color: #9CA3AF; margin-bottom: 4px; }
.src-snip  { font-size: 0.73rem; color: #6B7280; line-height: 1.5; }
.src-badge {
    display: inline-block;
    margin-top: 5px;
    font-size: 0.66rem;
    font-weight: 700;
    color: #10B981;
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 4px;
    padding: 1px 6px;
}

/* ── Follow-ups ── */
.fu-wrap { margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.05); }
.fu-label {
    font-size: 0.67rem; font-weight: 700; color: #374151;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;
}

/* ── Searching badge ── */
@keyframes shimmer {
    0%   { background-position: -600px 0; }
    100% { background-position:  600px 0; }
}
.searching-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 16px;
    background: linear-gradient(
        to right,
        rgba(255,106,0,0.04) 8%,
        rgba(255,106,0,0.18) 18%,
        rgba(255,106,0,0.04) 33%
    );
    background-size: 800px 100%;
    animation: shimmer 1.8s infinite linear;
    border: 1px solid rgba(255,106,0,0.22);
    border-radius: 100px;
    color: #FF6A00;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 18px;
    letter-spacing: 0.02em;
}

/* ── Chat input ── */
.stChatInputContainer {
    background: transparent !important;
    border: none !important;
    padding: 0 24px 26px 24px !important;
    max-width: 760px !important;
    margin: 0 auto !important;
    width: 100% !important;
}
div[data-testid="stChatInput"] {
    background: #111827 !important;
    border: 1px solid #374151 !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stChatInput"]:focus-within {
    border-color: rgba(255,106,0,0.45) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5), 0 0 0 3px rgba(255,106,0,0.07) !important;
}
div[data-testid="stChatInput"] textarea {
    color: #F3F4F6 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.94rem !important;
}
div[data-testid="stChatInput"] textarea::placeholder { color: #374151 !important; }
button[data-testid="stChatInputSubmitButton"] {
    background: linear-gradient(135deg, #FF6A00, #d95e00) !important;
    border-radius: 10px !important;
    padding: 7px !important;
    margin: 5px 7px !important;
    box-shadow: 0 2px 10px rgba(255,106,0,0.3) !important;
    transition: opacity 0.2s !important;
}
button[data-testid="stChatInputSubmitButton"] svg { fill: white !important; }
button[data-testid="stChatInputSubmitButton"]:hover { opacity: 0.85 !important; }

/* ── Mode popover ── */
div[data-testid="stPopover"] {
    position: fixed !important;
    bottom: 36px !important;
    left: 283px !important;
    z-index: 9999 !important;
    margin: 0 !important;
    width: 64px !important;
    height: 36px !important;
    pointer-events: none !important;
}
@media (max-width: 991px) {
    div[data-testid="stPopover"] { left: 46px !important; }
}
div[data-testid="stPopover"] button {
    pointer-events: auto !important;
    background: #1F2937 !important;
    background-color: #1F2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
    width: 64px !important;
    height: 36px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
    transition: all 0.15s !important;
}
div[data-testid="stPopover"] button:hover {
    background-color: #273344 !important;
    border-color: rgba(255,106,0,0.5) !important;
}
div[data-testid="stPopover"] button p {
    margin: 0 !important;
    line-height: 1 !important;
    font-size: 0.6rem !important;
    font-weight: 700 !important;
    color: #9CA3AF !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
div[data-testid="stPopover"] button svg,
div[data-testid="stPopover"] button span > svg { display: none !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: transparent !important;
    color: #4B5563 !important;
    font-size: 0.76rem !important;
    border: none !important;
}
.streamlit-expanderHeader:hover { color: #FF6A00 !important; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-wordmark">DEW<em>21</em></div>
        <div class="sb-logo-badge">AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-status">
        <div class="sb-dot"></div>
        System Online &nbsp;·&nbsp; RAG Active
    </div>
    """, unsafe_allow_html=True)

    if st.button(f"＋  {t['clear']}", use_container_width=True):
        if st.session_state.messages:
            save_chat(st.session_state.chat_id, generate_title(st.session_state.messages),
                      st.session_state.messages, st.session_state.sources)
        st.session_state.messages  = []
        st.session_state.sources   = []
        st.session_state.chat_id   = str(uuid.uuid4())
        st.session_state.chat_title = "New Chat"
        st.rerun()

    st.markdown('<div class="sb-section">History</div>', unsafe_allow_html=True)
    saved_chats = list_chats()
    for chat_meta in saved_chats:
        cid = chat_meta["id"]
        is_active = cid == st.session_state.chat_id
        col_t, col_d = st.columns([5, 1])
        with col_t:
            label = ("▶ " if is_active else "💬 ") + chat_meta["title"]
            if st.button(label, key=f"load_{cid}", use_container_width=True):
                if st.session_state.messages:
                    save_chat(st.session_state.chat_id,
                              generate_title(st.session_state.messages),
                              st.session_state.messages, st.session_state.sources)
                cd = load_chat(cid)
                if cd:
                    st.session_state.chat_id    = cd["id"]
                    st.session_state.chat_title = cd.get("title", "Untitled")
                    st.session_state.messages   = cd.get("messages", [])
                    st.session_state.sources    = cd.get("sources", [])
                    st.rerun()
        with col_d:
            if st.button("🗑", key=f"del_{cid}"):
                delete_chat(cid)
                if is_active:
                    st.session_state.messages  = []
                    st.session_state.sources   = []
                    st.session_state.chat_id   = str(uuid.uuid4())
                    st.session_state.chat_title = "New Chat"
                st.rerun()

    if not saved_chats:
        st.caption("No saved chats yet.")

    st.markdown('<div class="sb-section">Language</div>', unsafe_allow_html=True)
    lc1, lc2 = st.columns(2)
    with lc1:
        if st.button("🇺🇸 EN", use_container_width=True):
            st.session_state.lang = "en"
            st.rerun()
    with lc2:
        if st.button("🇩🇪 DE", use_container_width=True):
            st.session_state.lang = "de"
            st.rerun()


# =========================================================
# MAIN AREA
# =========================================================
st.markdown('<div class="main-wrap">', unsafe_allow_html=True)
chat_content = st.container()

with chat_content:
    if not st.session_state.messages:
        # ── HERO ──
        st.markdown(f"""
        <div class="hero-wrap">
            <div class="hero-eyebrow">⚡ DEW21 Energy Intelligence</div>
            <div class="hero-title">Smarter answers for<br><em>energy</em> questions.</div>
            <div class="hero-sub">{t['subtitle']}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── SUGGESTION CARDS ──
        st.markdown('<div class="cards-outer">', unsafe_allow_html=True)
        row1 = st.columns(2)
        row2 = st.columns(2)
        rows = [row1, row2]
        for i, ex in enumerate(t["examples"]):  # type: ignore
            col = rows[i // 2][i % 2]
            with col:
                st.markdown(f"""
                <div class="scard">
                    <div class="scard-icon">{ex['icon']}</div>
                    <div>
                        <div class="scard-title">{ex['title']}</div>
                        <div class="scard-sub">{ex['sub']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(ex["title"], key=f"ex_{i}", use_container_width=True, help=ex["sub"]):
                    st.session_state.example_trigger = ex["title"]
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # ── CHAT MESSAGES ──
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

        for i, msg in enumerate(st.session_state.messages):  # type: ignore
            role   = msg["role"]
            avatar = "⚡" if role == "assistant" else None
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg["content"])

                if role == "assistant":
                    # Source cards
                    if i < len(st.session_state.sources):
                        sdata = st.session_state.sources[i]
                        if sdata and sdata.get("highlights"):
                            with st.expander("📎 Sources & Citations"):
                                for h in sdata["highlights"]:
                                    st.markdown(f"""
                                    <div class="source-card">
                                        <div class="src-icon">📄</div>
                                        <div>
                                            <div class="src-name">{h['source']}</div>
                                            <div class="src-snip">{h['text']}</div>
                                            <span class="src-badge">✓ High Relevance</span>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)

                    # Follow-up suggestions (last assistant message only)
                    if i == len(st.session_state.messages) - 1:
                        st.markdown("""
                        <div class="fu-wrap">
                            <div class="fu-label">You may also ask</div>
                        </div>
                        """, unsafe_allow_html=True)
                        fu_cols = st.columns(len(t["followups"]))
                        for fi, fu in enumerate(t["followups"]):  # type: ignore
                            if fu_cols[fi].button(f"→ {fu}", key=f"fu_{i}_{fi}"):
                                st.session_state.example_trigger = fu
                                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# MODE POPOVER + CHAT INPUT
# =========================================================
with st.popover("MODE", use_container_width=False):
    st.markdown(
        "<p style='font-size:0.7rem;color:#4B5563;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.1em;margin-bottom:10px;'>Response Mode</p>",
        unsafe_allow_html=True
    )
    modes_list: list[dict[str, str]] = t["modes"]  # type: ignore
    for m in modes_list:
        is_active = st.session_state.mode == m["key"]
        prefix = "✓ " if is_active else ""
        if st.button(f"{prefix}{m['icon']} {m['key']}", key=f"mode_{m['key']}",
                     use_container_width=True, help=m["desc"]):
            st.session_state.mode = m["key"]
            st.rerun()

user_query = st.chat_input(t["placeholder"], disabled=st.session_state.generating)

if hasattr(st.session_state, "example_trigger") and st.session_state.example_trigger:
    user_query = st.session_state.example_trigger
    del st.session_state.example_trigger

if user_query:
    st.session_state.generating = True
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.sources.append(None)
    st.rerun()


# =========================================================
# GENERATION
# =========================================================
if (st.session_state.messages
        and st.session_state.messages[-1]["role"] == "user"
        and st.session_state.generating):

    last_query = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant", avatar="⚡"):
        with st.empty():
            st.markdown(
                '<div class="searching-badge">🔍&nbsp; Searching knowledge base...</div>',
                unsafe_allow_html=True
            )
            retrieved_docs    = []
            assistant_content = st.write_stream(
                ask_stream(
                    last_query,
                    chat_history=st.session_state.messages[:-1],
                    lang=st.session_state.lang,
                    retrieved_docs_out=retrieved_docs,
                    mode=st.session_state.mode,
                    doc_filter=st.session_state.context,
                )
            )

        highlights = []
        for d in retrieved_docs:
            source  = d.metadata.get("doc_name", "Unknown Document")
            snippet = d.page_content.replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            highlights.append({"text": snippet, "source": source})

    st.session_state.messages.append({"role": "assistant", "content": assistant_content})
    st.session_state.sources.append({"highlights": highlights})
    st.session_state.generating = False

    title = generate_title(st.session_state.messages)
    st.session_state.chat_title = title
    save_chat(st.session_state.chat_id, title,
              st.session_state.messages, st.session_state.sources)

    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)  # close main-wrap