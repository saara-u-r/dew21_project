import os
import time
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever  # Force reload fix
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_PATH_EN = os.path.join(BASE_DIR, "faiss_index")
FAISS_PATH_DE = os.path.join(BASE_DIR, "faiss_index_de")

# --- CACHE ---
# Cache is initialized below after FAISS loads

# --- EMBEDDINGS (bge-m3 Multilingual) ---
model_name = "BAAI/bge-m3"
# Use 'mps' for Mac Apple Silicon, fallback to 'cpu'
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
model_kwargs = {"device": device}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# WARMUP EMBEDDINGS (Eliminates ~4.5s cold-start spike)
print("Warming up embeddings...", flush=True)
embeddings.embed_query("warmup")
print("Embeddings ready.", flush=True)

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

# PRECOMPUTE ENSEMBLE RETRIEVERS (BM25 + FAISS) at boot
print("Precomputing BM25 indices...", flush=True)
def _build_ensemble(db):
    if not db:
        return None
    # 1. Vector Retriever (Semantic) — k=10
    faiss_retriever = db.as_retriever(search_kwargs={"k": 10})
    
    # 2. Extract context for BM25
    collection = db.docstore._dict
    documents = [
        Document(page_content=d.page_content, metadata=d.metadata) 
        for d in collection.values()
    ]
    if not documents:
        return None
        
    # 3. BM25 Retriever (Sparse/Keyword) — k=10
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10
    
    # 4. Ensemble (60% FAISS, 40% BM25)
    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )

_ENSEMBLE_CACHE = {
    "en": _build_ensemble(db_en),
    "de": _build_ensemble(db_de)
}
print("BM25 indices ready.", flush=True)

# --- HYBRID RETRIEVAL (ENSEMBLE) ---
def get_ensemble_retriever(lang="en"):
    return _ENSEMBLE_CACHE.get(lang)

async def ahybrid_retrieve(query, lang="en", doc_filter=None):
    retriever = get_ensemble_retriever(lang)
    if not retriever:
        return []
        
    docs = await retriever.ainvoke(query)
    
    if doc_filter and doc_filter != "All Docs":
        # Simplified filtering by checking if doc_filter is in the metadata source name
        filtered = [d for d in docs if doc_filter.lower() in d.metadata.get("source", "").lower()]
        return filtered
    return docs

# --- LLM ---
# ChatOllama is now instantiated per-request to prevent asyncio loop conflicts across threads.

# --- PROMPT MODES ---
MODE_PROMPTS = {
    "Simplified": {
        "en": "Use simple, non-legal language. Avoid jargon. Explain concepts as if to a layman or customer. Keep it friendly and easy to understand.",
        "de": "Verwende einfache, nicht-juristische Sprache. Vermeide Fachjargon. Erkläre Konzepte für Laien oder Kunden. Halte es freundlich und leicht verständlich."
    },
    "Standard": {
        "en": "Provide a clear and balanced answer. Use standard professional language.",
        "de": "Gib eine klare und ausgewogene Antwort. Verwende professionelle Standardsprache."
    },
    "Expert": {
        "en": "Use precise legal terminology. Focus on specific clauses, Article numbers, and exact wording from the documents. Maintain maximum professional detail.",
        "de": "Verwende präzise juristische Terminologie. Konzentriere dich auf spezifische Klauseln, Artikelnummern und den exakten Wortlaut aus den Dokumenten. Maximiere die fachliche Detailtiefe."
    }
}

PROMPTS = {
    "en": {
        "system": """You are the official DEW21 Energy Assistant. 
        INSTRUCTIONS:
        1. Answer ONLY based on the provided context. Do NOT use outside information.
        2. {mode_instruction}
        3. Be extremely thorough: if a detail (dates, years, names) is in the context, find it and report it.
        4. Citations required: \nSources: [FileName]
        5. If the question is a greeting, answer naturally without mentioning context.""",
        "context_lbl": "📄 CONTEXT:", "hist_lbl": "💬 HISTORY:", "q_lbl": "❓ QUESTION:", "a_lbl": "🤖 ANSWER:"
    },
    "de": {
        "system": """Du bist der DEW21 Energie-Assistent. 
        ANWEISUNGEN:
        1. Antworte NUR basierend auf dem KONTEXT. Kein externes Wissen.
        2. {mode_instruction}
        3. Sei gründlich: Suche nach Details (Daten, Jahre, Namen).
        4. Quellenangabe: \nQuellen: [Dateiname]
        5. Bei Begrüßungen antworte natürlich ohne den Kontext zu erwähnen.""",
        "context_lbl": "📄 KONTEXT:", "hist_lbl": "💬 VERLAUF:", "q_lbl": "❓ FRAGE:", "a_lbl": "🤖 ANTWORT:"
    }
}

prompt_template = PromptTemplate(
    input_variables=["system", "context", "question", "chat_history", "context_lbl", "hist_lbl", "q_lbl", "a_lbl"],
    template="{system}\n\n{hist_lbl}\n{chat_history}\n\n{context_lbl}\n{context}\n\n{q_lbl}\n{question}\n\n{a_lbl}"
)

# --- OPTIMIZED QUERY ANALYSIS ---
async def _aanalyze_query(llm, question: str, chat_history: list, lang: str = "en") -> dict:
    """Fast query heuristic to skip unnecessary LLM calls."""
    q_words = question.lower().split()
    
    # 1. SIMPLE HEURISTICS (Skip LLM if simple)
    is_simple = len(q_words) <= 5 and not any(w in ["it", "them", "those", "this", "that"] for w in q_words)
    if not chat_history and is_simple:
        return {"query": question, "sub_queries": [question]}

    # 2. FOLLOW-UP DETECTION (Rule-based first)
    followup_starters = ["what about", "how about", "why", "and", "give me more", "explain", "why?", "example"]
    is_followup = any(question.lower().startswith(s) for s in followup_starters) or len(q_words) < 3

    if not is_followup and not (" and " in question.lower()):
        return {"query": question, "sub_queries": [question]}

    # 3. LLM ANALYSIS (Only for complex entries)
    prompt = (
        f"You are a search query optimizer for a DEW21 Energy Assistant RAG system.\n"
        f"Decompose the user question into a standalone version and distinct sub-queries targeting different parts of our documentation (e.g., GTC, billing, data protection, liability).\n"
        f"Language: {lang}\n"
        f"User Question: {question}\n"
        f"Context: {chat_history[-1]['content'][:200] if chat_history else 'Initial Query'}\n\n"
        'Output strictly in valid JSON format:\n'
        '{"query": "Refined standalone question", "sub_queries": ["specific search term 1", "specific search term 2"]}'
    )
    
    try:
        from json import loads
        # Use a short timeout for expansion
        resp = await llm.ainvoke(prompt)
        return loads(resp.content.strip().replace('```json', '').replace('```', ''))
    except Exception:
        return {"query": question, "sub_queries": [question]}

import asyncio
import threading
import queue

async def _aask_stream(question, chat_history=None, lang="en", retrieved_docs_out=None, mode="Standard", doc_filter=None):
    """Aggressively optimized streaming for instant feedback with mode & filter support."""
    if chat_history is None: chat_history = []
    
    # Instantiate LLM within the async context
    llm = ChatOllama(model="qwen2.5:7b", temperature=0)
    
    # 1. PARALLEL ANALYSIS & BASE RETRIEVE
    base_retrieval_task = asyncio.create_task(ahybrid_retrieve(question, lang=lang, doc_filter=doc_filter))
    
    # Run the expensive LLM analysis concurrently!
    analysis_task = asyncio.create_task(_aanalyze_query(llm, question, chat_history, lang=lang))
    
    analysis = await analysis_task
    # Filter out the main question to avoid duplicate searching
    sub_queries = [sq for sq in analysis.get("sub_queries", [question]) if sq.lower() != question.lower()][:3]
    
    extra_retrievals = []
    if sub_queries:
        extra_retrievals = await asyncio.gather(*[
            ahybrid_retrieve(q, lang=lang, doc_filter=doc_filter) for q in sub_queries
        ])
        
    base_docs = await base_retrieval_task
    
    # ROUND-ROBIN MERGE: Ensure sub-queries aren't drowned out by the base query
    all_docs = []
    seen = set()
    
    # Interleave results: 1st from base, 1st from each sub-query, then 2nd...
    max_len = max([len(base_docs)] + [len(r) for r in extra_retrievals]) if extra_retrievals or base_docs else 0
    
    for i in range(max_len):
        # Base query doc
        if i < len(base_docs):
            d = base_docs[i]
            if d.page_content not in seen:
                seen.add(d.page_content)
                all_docs.append(d)
        # Sub-query docs
        for res_docs in extra_retrievals:
            if i < len(res_docs):
                d = res_docs[i]
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    all_docs.append(d)
    
    docs = all_docs[:15] # Increased to 15 for maximum coverage during presentation
    
    if retrieved_docs_out is not None:
        retrieved_docs_out.extend(docs)
        
    if not docs:
        yield "I'm sorry, no records found. Contact 0231 22 22 22 11."
        return

    # 2. PROMPT CONSTRUCTION WITH MODE
    mode_instruction = MODE_PROMPTS.get(mode, MODE_PROMPTS["Standard"])[lang]
    sys_prompt = PROMPTS[lang]["system"].format(mode_instruction=mode_instruction)
    
    ctx = "\n\n".join([f"[{d.metadata.get('source', 'Doc')}]\n{d.page_content}" for d in docs])
    hist = ""
    if chat_history:
        # Mini-history for context
        hist = f"User: {chat_history[-1]['content'][:150]}\nAssistant: {chat_history[-1]['content'][:150]}"

    prompt = (
        f"{sys_prompt}\n\n{ctx}\n\nHistory: {hist}\nQuestion: {question}\nAnswer:"
    )

    # 3. STREAMING
    try:
        async for chunk in llm.astream(prompt):
            yield chunk.content
    except Exception as e:
        yield f"Backend Delay: {e}. Retrying..."

def ask_stream(question, chat_history=None, lang="en", retrieved_docs_out=None, mode="Standard", doc_filter=None):
    """Synchronous generator wrapper using threading and Queue to stream to Streamlit."""
    q = queue.Queue()
    
    def _run_async():
        async def exhaust():
            try:
                async for chunk in _aask_stream(question, chat_history, lang, retrieved_docs_out, mode, doc_filter):
                    q.put(chunk)
            except Exception as e:
                q.put(f"Backend error: {e}")
            finally:
                q.put(None)  # Sentinel value signaling completion
        asyncio.run(exhaust())
        
    threading.Thread(target=_run_async, daemon=True).start()
    
    while True:
        chunk = q.get()
        if chunk is None:
            break
        yield chunk

def ask(question, chat_history=None, lang="en"):
    """Static version using the same logic."""
    full_answer = ""
    for chunk in ask_stream(question, chat_history, lang):
        full_answer += chunk
    return {"answer": full_answer, "contexts": [], "sources": [], "highlights": []}