import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIG (DE) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data_de")
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index_de")

# --- MODEL (bge-m3 Multilingual) ---
model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

def ingest_documents_de():
    print(f"📂 Using DATA_PATH (DE): {DATA_PATH}")
    
    # 🔥 CLEAR OLD INDEX
    if os.path.exists(FAISS_PATH):
        print(f"🗑️ Clearing old index at {FAISS_PATH} for fresh ingestion...")
        shutil.rmtree(FAISS_PATH)
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} does not exist. Please add German .txt files there.")
        return

    all_docs = []
    from langchain_core.documents import Document
    
    for f in os.listdir(DATA_PATH):
        if f.endswith(".txt"):
            print(f"📄 Loading DE TXT: {f}")
            fpath = os.path.join(DATA_PATH, f)
            with open(fpath, 'r', encoding='utf-8') as file:
                content = file.read()
                # Detailed metadata extraction
                doc_name = f.replace('.txt', '').replace('_de', '')
                all_docs.append(Document(
                    page_content=content, 
                    metadata={"source": f, "doc_name": doc_name, "lang": "de"}
                ))

    if not all_docs:
        print(f"❌ No documents found in {DATA_PATH} folder!")
        return

    # ✂️ SEMANTIC CHUNKING (Improved)
    print(f"✂️ Splitting {len(all_docs)} German sections into optimized Chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"✅ Total fine-grained DE chunks: {len(chunks)}")

    # 📦 STORE IN FAISS
    print("🧠 Creating bge-m3 embeddings and storing in FAISS Index (DE)...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_PATH)
    print(f"✨ FAISS Index (DE) ready and saved at {FAISS_PATH}!")

if __name__ == "__main__":
    ingest_documents_de()
