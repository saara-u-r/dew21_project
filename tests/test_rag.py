from src.rag import ask

while True:
    q = input("\nAsk: ")
    if q == "exit":
        break

    answer, docs = ask(q)

    print("\nANSWER:\n", answer)

    print("\nSOURCES:")
    for d in docs:
        print("-", d.metadata)