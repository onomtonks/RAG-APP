# RAG-PDF-Chroma

A **Retrieval-Augmented Generation (RAG) system** for PDFs using **ChromaDB**, **Python**, and **sentence embeddings**.  
This project extracts text (including images via OCR) from PDFs, creates hierarchical embeddings, and allows interactive querying.

---

## 🔹 Features

- Extract text from PDFs using **PyPDF**.
- Extract text from images inside PDFs using **pdf2image + pytesseract**.
- Hierarchical chunking using **Table of Contents (TOC)**.
- Store text chunks with metadata in **ChromaDB**.
- Interactive querying via CLI.
- Fully containerized with **Docker**.

---

## 🗂️ Folder Structure


---

## 📖 How It Works

### 1️⃣ High-Level Flow

```mermaid


flowchart TD
    A[PDF Page] --> B[Text Extraction with PyPDF]
    A --> C[Images Extraction with pdf2image + pytesseract]
    B --> D[Combine text from page]
    C --> D
    D --> E[Segment by TOC Sections]
    E --> F[Chunk each section]
    F --> G[Compute embeddings (SentenceTransformer)]
    G --> H[Store in ChromaDB (with metadata)]
    H --> I[Query Engine: input() or programmatic]
