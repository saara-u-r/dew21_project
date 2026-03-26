import sys, os
from src.rag import llm, hybrid_retrieve

q = "Are there any additional charges for late payment?"
prompt = f"Expand the following query with common synonyms and variations, especially formal legal terms that might appear in a German-to-English translated energy contract (like default, overdue, reimburse, costs). Output ONLY the expanded query string combining the original words and synonyms.\nQuery: {q}"
expanded = llm.invoke(prompt).content.strip()
print("Expanded:", expanded)

sq_docs = hybrid_retrieve(expanded, lang="en")
print(f"Retrieved {len(sq_docs)} for expanded")
for i, d in enumerate(sq_docs):
    src = d.metadata.get('doc_name', d.metadata.get('source', 'Doc'))
    print(f"--- Chunk {i+1} [{src}] ---")
    print(d.page_content[:200].replace('\n', ' '))
