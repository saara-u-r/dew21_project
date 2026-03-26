import sys
import os
import asyncio
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag import ask
from langchain_ollama import ChatOllama


async def main():
    query = "Compare Electricity vs Gas rights"
    print("Asking query...")
    ans = ask(query)

    print("--- ANSWER ---")
    print(ans["answer"])

asyncio.run(main())
