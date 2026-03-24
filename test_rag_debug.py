import sys, os
from src.rag import _is_vague_followup, _contextualize_query, _decompose_query, hybrid_retrieve

q = "I have few questions about my contract.... Are there any additional charges for late payment?"
print("Original:", q)
if _is_vague_followup(q):
    print("Is vague followup!")
    q = _contextualize_query(q, [{"role": "assistant", "content": "Hello! I am the DEW21 Energy Assistant. How can I help you today?"}])
    print("Contextualized:", q)
else:
    print("Not vague")

sub_queries = _decompose_query(q)
print("Subqueries:")
for sq in sub_queries:
    print(" - ", sq)

all_docs = []
seen_contents = set()
for sq in sub_queries:
    sq_docs = hybrid_retrieve(sq, lang="en")
    print(f"Retrieved {len(sq_docs)} for {sq}")
    for d in sq_docs:
        if d.page_content not in seen_contents:
            seen_contents.add(d.page_content)
            all_docs.append(d)

docs = all_docs[:8]
print(f"Total unique chunks kept: {len(docs)}")
for i, d in enumerate(docs):
    src = d.metadata.get('doc_name', d.metadata.get('source', 'Doc'))
    print(f"--- Chunk {i+1} [{src}] ---")
    print(d.page_content[:100].replace('\n', ' '))
