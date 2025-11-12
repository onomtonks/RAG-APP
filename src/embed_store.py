# src/embed_store.py
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .pdf_reader import extract_text_and_toc, segment_text_by_toc

def hierarchical_chunk_section(section):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    return splitter.split_text(section["text"])

def embed_pdf_text(pdf_path, client, collection_name="pdf_docs"):
    print(f"Processing {pdf_path} ...")

    full_text, page_texts, toc = extract_text_and_toc(pdf_path)
    print(f"Found {len(toc)} TOC entries")

    sections = segment_text_by_toc(page_texts, toc)
    print(f"Segmented into {len(sections)} logical sections")

    collection = client.get_or_create_collection(collection_name)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for sec in sections:
        chunks = hierarchical_chunk_section(sec)

        embeddings = model.encode(chunks).tolist()
        ids = [f"{sec['title']}-page-{sec['start_page']}-chunk-{i}" for i in range(len(chunks))]

        metadata = [{"section": sec["title"], "start_page": sec["start_page"]}] * len(chunks)

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata
        )

    print("✅ PDF embedded with TOC hierarchy.\n")
    return collection
