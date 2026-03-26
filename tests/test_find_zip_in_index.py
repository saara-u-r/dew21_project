import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
all_docs = db.docstore._dict
found = False
for k, v in all_docs.items():
    if "44135" in v.page_content:
        print(f"FOUND 44135 in {k}: {v.page_content[:500]}...")
        found = True

if not found:
    print("44135 NOT found in index")
