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
        
    # 1. Vector Retriever (Semantic)
    faiss_retriever = db.as_retriever(search_kwargs={"k": 12})
    
    # 2. Extract context for BM25
    collection = db.docstore._dict
    documents = [
        Document(page_content=d.page_content, metadata=d.metadata) 
        for d in collection.values()
    ]
    if not documents:
        return None
        
    # 3. BM25 Retriever (Sparse/Keyword)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 12
    
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
llm = ChatOllama(model="llama3", temperature=0)

# --- PROMPT CONFIG ---
PROMPTS = {
    "en": {
        "system": """You are the official DEW21 Energy Assistant. 
        Provide a helpful and accurate 2-3 line answer based STRICTLY on the provided documents.
        
        CRITICAL RULES:
        1. LANGUAGE: Always answer in English.
        2. NO PREAMBLES: Answer immediately.
        3. STRICT CONTEXT: Only answer if the information is EXPLICITLY in the documents. DO NOT generalize or use outside knowledge.
        4. NO EXTERNAL HELP: Do NOT suggest visiting the website, contacting customer support, or seeking external help.
        5. HONESTY: If the information is missing, say: "The provided documents do not contain instructions for this process." do not add any advice.
        6. LENGTH: Keep it between 2-3 lines.
        7. SOURCES: List document names at the end: "Sources: [Doc]".""",
        "context_lbl": "📄 CONTEXT:", "hist_lbl": "💬 HISTORY:", "q_lbl": "❓ QUESTION:", "a_lbl": "🤖 ANSWER:"
    },
    "de": {
        "system": """Du bist der offizielle DEW21 Energie-Assistent. 
        Gib eine hilfreiche und präzise Antwort in 2-3 Zeilen, die sich STRENG an die Dokumente hält.
        
        WICHTIGE REGELN:
        1. SPRACHE: Antworte AUSSCHLIESSLICH auf Deutsch. Auch wenn der Kontext auf Englisch ist, musst du ihn übersetzen.
        2. KEINE EINLEITUNG: Antworte sofort.
        3. STRENGER KONTEXT: Nutze NUR Informationen, die EXPLIZIT in den Dokumenten stehen. Nicht verallgemeinern.
        4. KEINE EXTERNEN HILFEN: Verweise NICHT auf die Website, den Kundenservice oder andere externe Hilfen.
        5. EHRLICHKEIT: Wenn Informationen fehlen, sage: "Die vorliegenden Dokumente enthalten keine Anleitung für diesen Vorgang." Gib keine weiteren Tipps.
        6. LÄNGE: Halte dich an 2-3 Zeilen.
        7. QUELLEN: Liste die Dokumente am Ende auf: "Quellen: [Doc]".""",
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

def ask(question, chat_history=None, lang="en"):
    if chat_history is None: chat_history = []
    c = PROMPTS.get(lang, PROMPTS["en"])
    
    # 1. RETRIEVE Hybrid Documents based on language
    docs = hybrid_retrieve(question, lang=lang)
    
    if not docs:
        fallback = "I'm sorry, I couldn't find specific documents for your query. Please contact DEW21 service at 0231 22 22 22 11."
        if lang == "de": fallback = "Es wurden keine passenden Dokumente gefunden. Bitte kontaktieren Sie DEW21 unter 0231 22 22 22 11."
        return {"answer": fallback, "contexts": [], "sources": [], "highlights": []}
    
    # 2. FORMAT CONTEXT
    ctx = "\n\n".join([f"Source: {d.metadata.get('doc_name', d.metadata.get('source','Doc'))} | {d.page_content}" for d in docs])
    hist = "\n".join([f"{m['role']}: {m['content'][:100]}..." for m in chat_history[-3:]])
    
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