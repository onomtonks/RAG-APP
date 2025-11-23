import os
import time
import socket # Needed for wait_for_chroma
from src.summarise import summary
#from src.chunking import chunks
from src.embed_store import embed_pdf_text
from src.query_engine import interactive_query
from chromadb import HttpClient 
import json# Correct import

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))

def wait_for_chroma(host, port, timeout=60):
    """Waits for the ChromaDB server to accept connections."""
    start = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"Chroma server is up at {host}:{port}")
                return
        except OSError:
            if time.time() - start > timeout:
                raise RuntimeError(f"Chroma server not available at {host}:{port} after {timeout}s")
            print("Waiting for Chroma server...")
            time.sleep(2)

def main():
    # 1. Wait for the server to start
    wait_for_chroma(CHROMA_HOST, CHROMA_PORT)

    # 2. Connect using HttpClient (Fixes the connection issue)
    client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    print(f"Connected to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}")


    # This step was crashing before due to missing 'pdfinfo' (poppler-utils)
    with open("srcdocker-compose run --rm app/final_chunks.json", "r") as f:
        json_chunks = json.load(f)
    final = embed_pdf_text(summary(json_chunks), client, collection_name="pdf_docs")
    interactive_query(final)
    

if __name__ == "__main__":
    main()