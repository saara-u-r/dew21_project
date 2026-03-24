import sys, os
from src.rag import _decompose_query, hybrid_retrieve

q = "What is the postalcode of DEW21?"
sub_queries = _decompose_query(q)
print(f"Original: {q}")
print(f"Sub-queries: {sub_queries}")

all_docs = []
seen_contents = set()
for sq in sub_queries:
    sq_docs = hybrid_retrieve(sq, lang="en")
    print(f"Retrieved {len(sq_docs)} for {sq}")
    for d in sq_docs:
        if d.page_content not in seen_contents:
            seen_contents.add(d.page_content)
            all_docs.append(d)

print(f"Top 3 unique chunks:")
for i, d in enumerate(all_docs[:3]):
    print(f"[{i}] {d.page_content[:200]}...")
