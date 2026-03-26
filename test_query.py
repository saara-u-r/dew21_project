import asyncio
import json
from src.rag import ahybrid_retrieve, _aanalyze_query, _aask_stream
from langchain_ollama import ChatOllama

async def main():
    llm = ChatOllama(model="qwen2.5:7b", temperature=0)
    query = "Compare Electricity vs Gas rights"
    
    analysis = await _aanalyze_query(llm, query, [])
    print("ANALYSIS:", json.dumps(analysis, ensure_ascii=False))
    
    docs = await ahybrid_retrieve(query)
    for i, d in enumerate(docs):
        print(f"--- DOC {i}: {d.metadata.get('source')} ---")
        print(d.page_content)

asyncio.run(main())
