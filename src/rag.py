import os
import time
from typing import Any
from langchain_community.retrievers import BM25Retriever  # type: ignore
from langchain_classic.retrievers import EnsembleRetriever  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from langchain_ollama import ChatOllama  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_core.prompts import PromptTemplate  # type: ignore
from langchain_core.documents import Document  # type: ignore
from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_PATH_EN = os.path.join(BASE_DIR, "faiss_index")
FAISS_PATH_DE = os.path.join(BASE_DIR, "faiss_index_de")

# --- CACHE ---
# Cache is initialized below after FAISS loads

# --- EMBEDDINGS (bge-m3 Multilingual) ---
model_name = "BAAI/bge-m3"
# Use 'mps' for Mac Apple Silicon, fallback to 'cpu'
import torch  # type: ignore
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

# Keyword triggers for boosting small/underrepresented documents
_KEYWORD_BOOST_MAP = {
    "cost": ["Cost_Overview", "cost overview", "DEW21_Cost"],
    "price": ["Cost_Overview", "cost overview", "DEW21_Cost"],
    "fee": ["Cost_Overview", "cost overview", "DEW21_Cost"],
    "reconnect": ["Cost_Overview", "cost overview", "DEW21_Cost"],
    "disconnect": ["Cost_Overview", "cost overview", "DEW21_Cost"],
    "reminder": ["Cost_Overview", "cost overview", "DEW21_Cost"],
    "invoice copy": ["Cost_Overview", "cost overview", "DEW21_Cost"],
    "kosten": ["Cost_Overview", "cost overview", "DEW21_Cost"],
    "preis": ["Cost_Overview", "cost overview", "DEW21_Cost"],
    "gebühr": ["Cost_Overview", "cost overview", "DEW21_Cost"],
    "schufa": ["Schufa", "SCHUFA", "Anhang_Schufa"],
    "creditreform": ["Creditreform", "creditreform"],
    "bonitätsprüfung": ["Schufa", "Creditreform"],
    "creditworthiness": ["Schufa", "Creditreform"],
}

def _get_keyword_boost_sources(query: str) -> list[str]:
    """Check if the query triggers keyword-based source boosting."""
    q_lower = query.lower()
    boost_sources = []
    for keyword, sources in _KEYWORD_BOOST_MAP.items():
        if keyword in q_lower:
            boost_sources.extend(sources)
    return list(set(boost_sources))

def _boost_docs_by_source(docs: list, boost_sources: list[str], db, lang="en") -> list:
    """Ensure documents from boosted sources appear in results."""
    if not boost_sources or not db:
        return docs
    
    # Check if any retrieved doc already comes from a boosted source
    existing_sources = set()
    for d in docs:
        src = d.metadata.get("source", "").lower()
        existing_sources.add(src)
    
    # Find if boost sources are missing
    missing_boost = []
    for bs in boost_sources:
        found = any(bs.lower() in s for s in existing_sources)
        if not found:
            missing_boost.append(bs)
    
    if not missing_boost:
        return docs  # Boost sources already present
    
    # Force-retrieve from the full docstore for missing sources
    collection = db.docstore._dict
    boosted_docs = []
    for doc in collection.values():
        src = doc.metadata.get("source", "").lower()
        for bs in missing_boost:
            if bs.lower() in src:
                boosted_docs.append(
                    Document(page_content=doc.page_content, metadata=doc.metadata)
                )
    
    # Prepend boosted docs (max 3) to ensure they appear in final context
    return [boosted_docs[i] for i in range(min(3, len(boosted_docs)))] + docs

async def ahybrid_retrieve(query, lang="en", doc_filter=None, k=10):
    retriever = get_ensemble_retriever(lang)
    if not retriever:
        return []
    
    # Dynamically update k if needed
    if k != 10:
        # Note: This is an expensive change if done per query in production,
        # but for evaluation/parameter sweeps, it is necessary.
        for r in retriever.retrievers:
            if hasattr(r, "k"):
                setattr(r, "k", k)
            elif hasattr(r, "search_kwargs"):
                kwargs = getattr(r, "search_kwargs")
                if isinstance(kwargs, dict):
                    kwargs["k"] = k
        
    docs = await retriever.ainvoke(query)
    
    # KEYWORD BOOST: Force-include docs from small/underrepresented sources
    boost_sources = _get_keyword_boost_sources(query)
    if boost_sources:
        db = db_en if lang == "en" else db_de
        docs = _boost_docs_by_source(docs, boost_sources, db, lang)
    
    if doc_filter and doc_filter != "All Docs":
        filtered = [d for d in docs if doc_filter.lower() in d.metadata.get("source", "").lower()]
        return filtered
    return docs

# --- LLM ---
# ChatOllama is now instantiated per-request to prevent asyncio loop conflicts across threads.

# --- PROMPT MODES ---
MODE_PROMPTS = {
    "Simplified": {
        "en": "MODE: SIMPLIFIED. You MUST rephrase the context into very simple, everyday language. Avoid all legal jargon. Explain concepts patiently as if to a layman or customer. Be extremely friendly and helpful.",
        "de": "MODE: SIMPLIFIED. Du MUSST den Kontext in sehr einfache Alltagssprache umformulieren. Vermeide jeden Fachjargon. Erkläre Konzepte geduldig wie für einen Laien oder Kunden. Sei extrem freundlich und hilfsbereit."
    },
    "Standard": {
        "en": "MODE: STANDARD. Provide a clear, professional, and balanced answer. The answer should be slightly more detailed than the simplified mode.",
        "de": "MODE: STANDARD. Gib eine klare, professionelle und ausgewogene Antwort. Die Antwort sollte etwas detaillierter sein als im vereinfachten Modus."
    },
    "Expert": {
        "en": "MODE: EXPERT. You MUST use precise legal terminology. Visually break down the answer into structured bullet points. You MUST explicitly cite specific clauses, Article numbers, and exact wording from the context within your answer text. Give a detailed answer with a mix of paragraphs and bullet points.",
        "de": "MODE: EXPERT. Du MUSST präzise juristische Terminologie verwenden. Gliedere die Antwort visuell in strukturierte Aufzählungspunkte. Du MUSST explizit spezifische Klauseln, Artikelnummern und exakte Formulierungen aus dem Kontext innerhalb deines Antworttextes zitieren. Gib eine detaillierte Antwort mit einer Mischung aus Absätzen und Aufzählungspunkten."
    }
}

PROMPTS = {
    "en": {
        "system": """You are the official DEW21 Energy Assistant.

YOUR TONE AND FORMATTING MODE:
{mode_instruction}

STRICT GROUNDING RULES — FOLLOW THESE EXACTLY:
1. Use ONLY information that is explicitly found in the CONTEXT below.
2. Do NOT add fabricated examples, comparisons, or factual details beyond what the context states.
3. You may rephrase the text to match your assigned MODE, but you must NOT introduce new factual meaning or implications.
4. Every fact in your answer MUST be directly traceable to a specific passage in the context.
5. NEVER use your training knowledge to fill factual gaps.
6. If the context contains ANY information related to the question (even abstract rights like BGB or GDPR), you MUST provide that information instead of refusing.
7. If the context is broadly unrelated to the specific question, do NOT extract entirely unrelated rules. Briefly summarize whatever closely matches the topic, noting the exact answer is absent.
8. If the user asks for "the company's" contact details or "SC" company, ALWAYS provide the details for both DEW21 AND the dispute resolution body (Schlichtungsstelle Energie e. V.).
9. End your final output exactly with: Sources: [list the document filenames used]
10. If the question is a greeting, respond naturally without referencing documents."""
    },
    "de": {
        "system": """Du bist der offizielle DEW21 Energie-Assistent.

DEINE TON- UND FORMATIERUNGSANWEISUNGEN:
{mode_instruction}

STRIKTE GRUNDIERUNGSREGELN — BEFOLGE DIESE GENAU:
1. Verwende NUR Informationen, die explizit im KONTEXT unten stehen.
2. Füge KEINE erfundenen Beispiele, Vergleiche oder faktischen Details hinzu.
3. Du darfst den Text umformulieren, um deinem MODUS zu entsprechen, darfst aber KEINE neuen sachlichen Implikationen einführen.
4. Jede Tatsache in deiner Antwort MUSS direkt auf eine bestimmte Stelle im Kontext zurückführbar sein.
5. Verwende NIEMALS dein Trainingswissen, um faktische Lücken zu füllen.
6. Wenn der Kontext IRGENDWELCHE Informationen enthält, die mit der Frage zusammenhängen, MUSST du diese bereitstellen, anstatt abzulehnen.
7. Wenn der Kontext weitgehend unabhängig von der spezifischen Frage ist, extrahiere KEINE völlig unzusammenhängenden Regeln. Fasse kurz das Thema zusammen und merke an, dass die genaue Antwort fehlt.
8. Wenn nach Kontaktdaten gefragt wird, gib IMMER die Daten sowohl für DEW21 ALS AUCH für die Schlichtungsstelle Energie e. V. an.
9. Beende deine endgültige Antwort exakt mit: Quellen: [verwendete Dateinamen]
10. Bei Begrüßungen antworte natürlich ohne Dokumente zu erwähnen."""
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

    complexity_triggers = [" and ", " vs ", " vs. ", " or ", "compare", "difference", "both", "between"]
    is_complex = any(trigger in question.lower() for trigger in complexity_triggers)

    # Manual intercept for user commonly asking for the "sc something" company
    if "sc something" in question.lower() or ("contact" in question.lower() and "sc" in question.lower()):
        return {"query": question, "sub_queries": [question, "Schlichtungsstelle Energie e. V."]}

    if not is_followup and not is_complex:
        return {"query": question, "sub_queries": [question]}

    # 3. LLM ANALYSIS (Only for complex entries)
    ctx_str = "Initial Query"
    if len(chat_history) >= 2:
        ctx_str = f"User asked: {chat_history[-2]['content'][:150]} | Assistant answered: {chat_history[-1]['content'][:150]}"
    elif len(chat_history) == 1:
        ctx_str = f"User asked: {chat_history[-1]['content'][:150]}"

    prompt = (
        f"You are a search query optimizer for a DEW21 Energy Assistant RAG system.\n"
        f"Decompose the user question into a standalone version and distinct sub-queries targeting different parts of our documentation (e.g., GTC, billing, data protection, liability).\n"
        f"If the question compares two subjects (e.g., 'Compare Electricity vs Gas rights'), your sub_queries must uniquely split them (e.g., ['Electricity rights', 'Gas rights']).\n"
        f"Language: {lang}\n"
        f"User Question: {question}\n"
        f"Context: {ctx_str}\n\n"
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

async def _aask_stream(question: str, chat_history: list[dict[str, Any]] | None = None, lang: str = "en", retrieved_docs_out: list[Any] | None = None, mode: str = "Standard", doc_filter: str | None = None, k: int = 10):
    """Aggressively optimized streaming for instant feedback with mode & filter support."""
    if chat_history is None: chat_history = []
    
    # Instantiate LLM within the async context
    llm = ChatOllama(model="qwen2.5:7b", temperature=0)
    
    # 1. PARALLEL ANALYSIS & BASE RETRIEVE
    base_retrieval_task = asyncio.create_task(ahybrid_retrieve(question, lang=lang, doc_filter=doc_filter, k=k))
    
    # Run the expensive LLM analysis concurrently!
    analysis_task = asyncio.create_task(_aanalyze_query(llm, question, chat_history, lang=lang))
    
    analysis = await analysis_task
    # Filter out the main question to avoid duplicate searching
    raw_sub_queries = [sq for sq in analysis.get("sub_queries", [question]) if sq.lower() != question.lower()]
    sub_queries = [raw_sub_queries[i] for i in range(min(3, len(raw_sub_queries)))]
    
    extra_retrievals: list[list[Any]] = []
    if sub_queries:
        extra_retrievals_raw = await asyncio.gather(*[
            ahybrid_retrieve(q, lang=lang, doc_filter=doc_filter, k=k) for q in sub_queries
        ])
        extra_retrievals = list(extra_retrievals_raw)  # type: ignore
        
    base_docs: list[Any] = await base_retrieval_task  # type: ignore
    
    # ROUND-ROBIN MERGE with SOURCE DIVERSITY: Ensure sub-queries and
    # different document sources aren't drowned out by a single dominant doc.
    all_docs: list[Any] = []
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
    
    # SOURCE DIVERSITY: Ensure we don't have >10 chunks from the same source
    # This prevents a single large doc from monopolizing the context window
    source_counts = {}
    diverse_docs = []
    overflow_docs = []
    MAX_PER_SOURCE = 8
    
    for d in all_docs:
        src = d.metadata.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
        if source_counts[src] <= MAX_PER_SOURCE:
            diverse_docs.append(d)
        else:
            overflow_docs.append(d)
    
    # Fill remaining slots with overflow if we have room
    combined = diverse_docs + overflow_docs
    docs = [combined[i] for i in range(min(15, len(combined)))]
    
    if isinstance(retrieved_docs_out, list):
        retrieved_docs_out.extend(docs)
        
    if not docs:
        yield "I'm sorry, no records found."
        return

    # 2. PROMPT CONSTRUCTION WITH MODE
    mode_instruction = MODE_PROMPTS.get(mode, MODE_PROMPTS["Standard"])[lang]
    sys_prompt = PROMPTS[lang]["system"].format(mode_instruction=mode_instruction)
    
    ctx = "\n\n".join([f"[{d.metadata.get('source', 'Doc')}]\n{d.page_content}" for d in docs])
    hist = ""
    if len(chat_history) >= 2:
        hist = f"User: {chat_history[-2]['content'][:150]}\nAssistant: {chat_history[-1]['content'][:150]}"
    elif len(chat_history) == 1:
        hist = f"User: {chat_history[-1]['content'][:150]}"

    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"CONTEXT:\n{ctx}\n\nHISTORY:\n{hist}\n\nQUESTION: {question}")
    ]

    # 3. STREAMING
    try:
        async for chunk in llm.astream(messages):
            yield chunk.content
    except Exception as e:
        yield f"Backend Delay: {e}. Retrying..."

def ask_stream(question, chat_history=None, lang="en", retrieved_docs_out=None, mode="Standard", doc_filter=None, k=10):
    """Synchronous generator wrapper using threading and Queue to stream to Streamlit."""
    q: queue.Queue[Any] = queue.Queue()
    
    def _run_async():
        async def exhaust():
            try:
                async for chunk in _aask_stream(question, chat_history, lang, retrieved_docs_out, mode, doc_filter, k=k):
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

def ask(question, chat_history=None, lang="en", mode="Standard", doc_filter=None, k=10):
    """Static version using the same logic."""
    full_answer = ""
    for chunk in ask_stream(question, chat_history=chat_history, lang=lang, mode=mode, doc_filter=doc_filter, k=k):
        if chunk is not None:
            full_answer += str(chunk)
    return {"answer": full_answer, "contexts": [], "sources": [], "highlights": []}