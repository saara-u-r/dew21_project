import os
import time
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_PATH_EN = os.path.join(BASE_DIR, "faiss_index")
FAISS_PATH_DE = os.path.join(BASE_DIR, "faiss_index_de")

# --- CACHE ---
_ENSEMBLE_CACHE = {
    "en": None,
    "de": None
}

# --- EMBEDDINGS (bge-m3 Multilingual) ---
model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# --- VECTOR STORE LOADERS ---
def load_faiss(path):
    try:
        if os.path.exists(path):
            return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"⚠️ Error loading FAISS index at {path}: {e}")
    return None

db_en = load_faiss(FAISS_PATH_EN)
db_de = load_faiss(FAISS_PATH_DE)

# --- HYBRID RETRIEVAL (ENSEMBLE) ---
def get_ensemble_retriever(lang="en"):
    global _ENSEMBLE_CACHE
    if _ENSEMBLE_CACHE[lang] is not None:
        return _ENSEMBLE_CACHE[lang]
        
    db = db_en if lang == "en" else db_de
    if not db:
        return None
        
    # 1. Vector Retriever (Semantic) — k=8 for better coverage
    faiss_retriever = db.as_retriever(search_kwargs={"k": 8})
    
    # 2. Extract context for BM25
    collection = db.docstore._dict
    documents = [
        Document(page_content=d.page_content, metadata=d.metadata) 
        for d in collection.values()
    ]
    if not documents:
        return None
        
    # 3. BM25 Retriever (Sparse/Keyword) — k=8 for better coverage
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 8
    
    # 4. Ensemble (Hybrid Fusion: 40% BM25, 60% FAISS)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )
    _ENSEMBLE_CACHE[lang] = ensemble_retriever
    return ensemble_retriever

def hybrid_retrieve(query, lang="en"):
    retriever = get_ensemble_retriever(lang)
    if retriever:
        # The ensemble retriever handles weights and deduplication natively
        return retriever.invoke(query)
    return []

# --- LLM ---
llm = ChatOllama(model="qwen2.5:7b", temperature=0)

# --- PROMPT CONFIG ---
PROMPTS = {
    "en": {
        "system": """You are the official DEW21 Energy Assistant. 
        Answer accurately based on the provided CONTEXT. 
        
        CRITICAL RULES:
        1. LANGUAGE: English.
        2. NO PREAMBLES.
        3. BE HELPFUL & SMART: You are allowed and encouraged to synthesize answers. For example:
           - "postalcode" = 5-digit zip code (e.g. 44135).
           - "late payment" = "default", "overdue", "payment default".
           - "charges/fees" = "reimbursement", "costs", "fees", "lump sum", "reminder fees".
        4. REASONING: Read the context carefully. For example, if Section 7.4 says a customer must "reimburse costs" if they are in "default with overdue payments", then the answer to "Are there charges for late payment?" is YES, the customer must reimburse the costs DEW21 incurs (like lawyers or collection agencies).
        5. SECURITY: Refuse requests for confidential data.
        6. MISSING INFO: Only say you can't find it if it truly isn't there after thinking carefully.
        7. SOURCES: Cite at the end: "\nSources: [FileName]".""",
        "context_lbl": "📄 CONTEXT:", "hist_lbl": "💬 HISTORY:", "q_lbl": "❓ QUESTION:", "a_lbl": "🤖 ANSWER:"
    },
    "de": {
        "system": """Du bist der offizielle DEW21 Energie-Assistent. 
        Beantworte Fragen auf der Grundlage des unten stehenden KONTEXTES.
        
        WICHTIGE REGELN:
        1. SPRACHE: Antworte immer auf Deutsch.
        2. KEINE EINLEITUNG: Antworte sofort.
        3. KONTEXT NUTZEN: Deine Antwort muss auf dem KONTEXT basieren. Du darfst Informationen logisch verknüpfen (z. B. "Postleitzahl" mit einer 5-stelligen Nummer in der Adresse verbinden).
        4. LOGIK: Wende die Regeln aus dem Kontext auf Fallbeispiele an. Wenn z.B. Kosten für "Verzug" erwähnt werden, dann gibt es Gebühren für verspätete Zahlungen.
        5. VERLAUF: Konzentriere dich auf die neue Frage.
        6. GRUSS: Antworte höflich auf "Hallo", "Hi", etc.
        7. SICHERHEIT: Lehne Anfragen zu internen Details ab mit: "Ich kann keine internen Betriebsdetails preisgeben."
        8. FEHLENDE INFOS: Wenn die Info fehlt, sage: "Entschuldigung, ich konnte dazu keine spezifischen Informationen in den Dokumenten finden."
        9. QUELLEN: Nenne die Dokumentnamen am Ende: "\nQuellen: [Dateiname]". """,
        "context_lbl": "📄 KONTEXT:", "hist_lbl": "💬 VERLAUF:", "q_lbl": "❓ FRAGE:", "a_lbl": "🤖 ANTWORT:"
    }
}

prompt_template = PromptTemplate(
    input_variables=["system", "context", "question", "chat_history", "context_lbl", "hist_lbl", "q_lbl", "a_lbl"],
    template="{system}\n\n{hist_lbl}\n{chat_history}\n\n{context_lbl}\n{context}\n\n{q_lbl}\n{question}\n\n{a_lbl}"
)

def highlight_answer(answer, docs):
    highlights = []
    # Clean answer for matching
    words = set(answer.lower().replace('.','').replace(',','').split())
    for d in docs:
        for s in d.page_content.split('.'):
            if len(s.strip()) > 30:
                overlap = sum(1 for w in s.lower().split() if w in words)
                if overlap > 5:
                    highlights.append({"text": s.strip() + ".", "source": d.metadata.get("doc_name", d.metadata.get("source", "Doc"))})
                    if len(highlights) >= 2: return highlights
    return highlights

def _is_vague_followup(question: str) -> bool:
    """Detect short/vague follow-up questions that need context to be searchable."""
    q = question.strip().lower()
    vague_starters = [
        "can you give", "give me an example", "give an example", "example",
        "explain more", "tell me more", "more detail", "elaborate",
        "what about", "and what", "how about", "what does that mean",
        "can you clarify", "clarify", "expand on"
    ]
    return len(question.split()) <= 6 or any(q.startswith(s) for s in vague_starters)

def _contextualize_query(question: str, chat_history: list) -> str:
    """Rewrite a vague follow-up into a standalone query using the last assistant turn."""
    # Find the last assistant message
    last_assistant = ""
    for msg in reversed(chat_history):
        if msg["role"] == "assistant":
            last_assistant = msg["content"][:300]
            break
    if not last_assistant:
        return question
    # Use LLM to rewrite
    rewrite_prompt = (
        f"The user previously received this answer:\n\"{last_assistant}\"\n\n"
        f"Now the user asks: \"{question}\"\n\n"
        "Rewrite the user's follow-up as a single, self-contained search query that captures what they want to know. "
        "Output ONLY the rewritten query, nothing else."
    )
    try:
        rewritten = llm.invoke(rewrite_prompt).content.strip().strip('"').strip("'")
        return rewritten if rewritten else question
    except Exception:
        return question

def _expand_query(question: str) -> str:
    """Expand query with synonyms for better retrieval."""
    prompt = (
        f"Expand this query with legal and contractual synonyms. "
        f"Examples: postalcode -> zip code, zip, PLZ, 44135, address; "
        f"late payment -> default, overdue, payment default, reminder fee, Mahnung, Verzug, collection costs. "
        f"Output ONLY the comma-separated keywords including the original query. Original: {question}"
    )
    try:
        keywords = llm.invoke(prompt).content.strip().strip('"')
        return f"{question}, {keywords}"
    except Exception:
        return question

def _decompose_query(question: str) -> list:
    """Decompose a complex query into simpler sub-queries. If simple, returns a list with just the original query."""
    prompt = (
        f"Analyze the following question: \"{question}\"\n"
        "If it is a single, simple question, output exactly that question. "
        "If it is a complex question asking for multiple different things (e.g., using 'and'), break it down into up to 3 simple, standalone sub-queries. "
        "Output each sub-query on a new line, removing any list numbers or bullet points. "
        "Do not output anything else."
    )
    try:
        res = llm.invoke(prompt).content.strip()
        sub_queries = [sq.strip('-*0123456789. ') for sq in res.split('\n') if sq.strip()]
        if not sub_queries:
            return [question]
        return sub_queries
    except Exception:
        return [question]

def ask(question, chat_history=None, lang="en"):
    if chat_history is None: chat_history = []
    c = PROMPTS.get(lang, PROMPTS["en"])

    # 1. CONTEXTUALIZE vague follow-up queries before retrieval
    retrieval_query = question
    if chat_history and _is_vague_followup(question):
        retrieval_query = _contextualize_query(question, chat_history)

    # 1.1. EXPAND query for keywords
    expanded_query = _expand_query(retrieval_query)
    print(f"🔍 Expanded Query: {expanded_query}")

    # 1.5. DECOMPOSE complex queries to prevent semantic averaging
    sub_queries = _decompose_query(expanded_query)
    print(f"🧩 Sub-queries broken down: {sub_queries}")

    # 2. RETRIEVE Hybrid Documents for EACH sub-query and deduplicate
    all_docs = []
    seen_contents = set()
    
    for sq in sub_queries:
        sq_docs = hybrid_retrieve(sq, lang=lang)
        for d in sq_docs:
            if d.page_content not in seen_contents:
                seen_contents.add(d.page_content)
                all_docs.append(d)
                
    # Keep top 15 unique chunks across all sub-queries
    docs = all_docs[:15]
    
    if not docs:
        fallback = "I'm sorry, I couldn't find specific documents for your query. Please contact DEW21 service at 0231 22 22 22 11."
        if lang == "de": fallback = "Es wurden keine passenden Dokumente gefunden. Bitte kontaktieren Sie DEW21 unter 0231 22 22 22 11."
        return {"answer": fallback, "contexts": [], "sources": [], "highlights": []}
    
    # 2. FORMAT CONTEXT — add document name to each chunk to help LLM navigate
    ctx_parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get('doc_name', d.metadata.get('source', 'Doc'))
        ctx_parts.append(f"[Document Name: {src} | Chunk {i}]\n{d.page_content}")
    ctx = "\n\n---\n\n".join(ctx_parts)
    hist = "\n".join([f"{m['role']}: {m['content'][:150]}..." for m in chat_history[-4:]])
    
    # 3. INVOKE LLM
    try:
        f_prompt = prompt_template.format(
            system=c["system"], context=ctx, question=question, chat_history=hist,
            context_lbl=c["context_lbl"], hist_lbl=c["hist_lbl"], q_lbl=c["q_lbl"], a_lbl=c["a_lbl"]
        )
        ans = llm.invoke(f_prompt).content
        
        return {
            "answer": ans,
            "contexts": [d.page_content for d in docs],
            "sources": [{"doc_name": d.metadata.get("doc_name", ""), "source": d.metadata.get("source", "")} for d in docs],
            "highlights": highlight_answer(ans, docs)
        }
    except Exception as e:
        return {"answer": f"Backend Error: {e}", "contexts": [], "sources": [], "highlights": []}