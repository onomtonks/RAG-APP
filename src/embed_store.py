# src/embed_store.py
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
# from langchain_text_splitters import RecursiveCharacterTextSplitter # Not used, so commented out
from typing import List, Dict, Any # Recommended for better type hinting
import json

# Initialize the model once outside the function to avoid redundant loading
# This makes the function much faster, especially if called multiple times.
try:
    # Use a global variable for the model (or pass it as an argument/use a class)
    # Using a global variable here for simplicity, assuming this module is imported once.
    EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    EMBEDDING_MODEL = None # Handle case where model might fail to load

def embed_pdf_text(
    # Updated type hint for clarity: expecting List[Dict] from the summary function
    json_data: List[Dict[str, Any]], 
    client: HttpClient, 
    collection_name: str = "pdf_docs"
):
    """
    Embeds text chunks into a ChromaDB collection using a Sentence Transformer model.
    Assumes json_data is a list of dictionaries, where each dict contains 
    'cluster' (list of IDs) and 'summary' (list of document chunks).
    """
    if EMBEDDING_MODEL is None:
        print("❌ Embedding model failed to load. Cannot proceed.")
        return None

    print("Processing embedding...")

    # 1. Get or Create Collection
    collection = client.get_or_create_collection(collection_name)
    
    # 2. Process and Embed Data
    for sec in json_data:
        # Check that 'summary' is a string and 'cluster' is present (INT or STR)
        if not isinstance(sec.get("summary"), str) or sec.get("cluster") is None:
            # Customizing the error message to help debug structure issues
            print(f"⚠️ Skipping section due to invalid structure: Summary type is {type(sec.get('summary'))}, Cluster is {sec.get('cluster')}")
            continue

        # Extract the single values
        single_chunk_text = sec["summary"]
        single_id = str(sec["cluster"]) # **CRITICAL: Convert ID to string**
        
        # Wrap the single items into lists of length 1 for ChromaDB and SentenceTransformer
        chunks = [single_chunk_text] # Must be a list of documents
        ids = [single_id]            # Must be a list of string IDs

        # Generate embeddings
        # The model expects a list of strings (chunks)
        embeddings = EMBEDDING_MODEL.encode(chunks, show_progress_bar=False).tolist()
        
        # 3. Add to Collection
        try:
            # ChromaDB requires lists for IDs, documents, and embeddings
            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
            )
        except Exception as e:
            print(f"❌ Error adding data to ChromaDB for cluster {single_id}: {e}")        
    print("✅ PDF embedded with TOC hierarchy.\n")
    return collection


if __name__ == "__main__":
    with open("src/chunk_summaries.json", "r") as f:
        json_chunks = json.load(f)

