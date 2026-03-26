import os
from src.rag import prompt_template, PROMPTS, llm, hybrid_retrieve, _decompose_query

q = "What is the postalcode of DEW21?"
c = PROMPTS["en"]

# Simulate the ask function logic
sub_queries = _decompose_query(q)
all_docs = []
seen_contents = set()
for sq in sub_queries:
    sq_docs = hybrid_retrieve(sq, lang="en")
    for d in sq_docs:
        if d.page_content not in seen_contents:
            seen_contents.add(d.page_content)
            all_docs.append(d)
docs = all_docs[:8]

ctx_parts = []
for i, d in enumerate(docs, 1):
    src = d.metadata.get('doc_name', d.metadata.get('source', 'Doc'))
    ctx_parts.append(f"[Document Name: {src} | Chunk {i}]\n{d.page_content}")
ctx = "\n\n---\n\n".join(ctx_parts)
hist = "" # empty for test

f_prompt = prompt_template.format(
    system=c["system"], context=ctx, question=q, chat_history=hist,
    context_lbl=c["context_lbl"], hist_lbl=c["hist_lbl"], q_lbl=c["q_lbl"], a_lbl=c["a_lbl"]
)

print("-" * 20)
print("PROMPT:")
# print(f_prompt) # No, too long.

ans = llm.invoke(f_prompt).content
print("-" * 20)
print("ANSWER:")
print(ans)
