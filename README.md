📘 RAG for AI Students

A Retrieval-Augmented Generation system for studying AI using textbook-based contextual answers.

📚 Overview

RAG for AI Students is a personalized Retrieval-Augmented Generation (RAG) pipeline designed to help students study AI more effectively.
The system extracts information from a Large Language Model (LLM) textbook (PDF), organizes it into meaningful chunks, and uses an LLM to answer questions grounded in the textbook content.

This allows you to:

Ask natural language questions about AI topics

Receive answers generated with actual book context

Use the system as a study assistant for revision, summaries, and explanations

✨ Features

PDF ingestion and dividing text into paragraphs

Embedding generation using Sentence Transformers on paragraphs

Dimensionality reduction using PCA

Clustering using HDBSCAN for context grouping

converting each cluster into a single chunks

Vector storage of each chunk and retrieval using ChromaDB

Query processing using LangChain + LLMs

Accurate, context-rich answers grounded in textbook content

OCR support for scanned PDFs


               +---------------------+
               |      PDF Input      |
               +----------+----------+
                          |
                          v
            +---------------------------+
            |   Paragraph Chunking      |
            | (based on '\n' markers)   |
            +-------------+-------------+
                          |
                          v
              +---------------------+
              |  Sentence Embedding |
              | (SentenceTransformer)|
              +-----------+----------+
                          |
                          v
                 +-----------------+
                 |   PCA Reduction |
                 +--------+--------+
                          |
                          v
                +--------------------+
                |    HDBSCAN Clustering |
                +---------+----------+
                          |
                          v
             +-----------------------------+
             |   ChromaDB Vector Storage   |
             +--------------+--------------+
                          |
                          v
               +----------------------+
               |  Query + Retrieval   |
               +----------+-----------+
                          |
                          v
                +----------------------+
                |     LLM Response     |
                +----------------------+



🧰 Tech Stack
Languages & Runtime

Python 3.11-slim

Core Libraries

numpy

chromadb

ML / NLP

sentence-transformers

sklearn (PCA)

hdbscan

LLM Framework

langchain

langchain-community

langchain-groq

langchain-openai

tiktoken

PDF / OCR

pdfminer.six

pytesseract

pdf2image

opencv-python

Pillow

lxml

Environment

python-dotenv


