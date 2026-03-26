import asyncio
import json
from src.rag import ask
from langchain_ollama import ChatOllama

async def main():
    query = "Compare Electricity vs Gas rights"
    import sys
    sys.path.insert(0, ".")
    print("Asking query...")
    ans = ask(query)
    print("--- ANSWER ---")
    print(ans["answer"])

asyncio.run(main())
