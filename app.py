import sys
from src.query_engine import interactive_query
from src.embed_store import embed_pdf_text
from chromadb import HttpClient

def main():
    client = HttpClient(host="chroma", port=8000)
    pdf_files = ["pdfs/llm-book.pdf"]

    for pdf in pdf_files:
        collection = embed_pdf_text(pdf, client)

    if sys.stdin.isatty():
        # Interactive mode
        interactive_query(collection)
    else:
        # Non-interactive mode (default query)
        query = "Summarize the book"
        results = collection.query(query_texts=[query], n_results=3)
        for i, doc in enumerate(results["documents"][0]):
            print(f"\n--- Result {i+1} ---\n{doc}")

if __name__ == "__main__":
    main()
