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
results = db.similarity_search("postal code address DEW21 zip code", k=5)

print(f"Results for address search:")
for i, res in enumerate(results):
    print(f"[{i}] {res.page_content[:200]}...")
