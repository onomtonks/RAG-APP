# src/query_engine.py
def interactive_query(collection):
    print("\nYou can now query the PDFs. Type 'exit' to quit.")

    while True:
        try:
            query = input("\nEnter your question: ")
        except EOFError:
            print("\n[Exited interactive mode]")
            break

        if query.lower() == "exit":
            break

        results = collection.query(query_texts=[query], n_results=3)
        for i, doc in enumerate(results["documents"][0]):
            print(f"\n--- Result {i+1} ---\n{doc}")
