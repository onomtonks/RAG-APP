"""
Semantic Chunking by Subtopics with BERT Embeddings
"""

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer
import chromadb
import numpy as np
from typing import List, Dict, Tuple
import fitz  # PyMuPDF
import re


class SubtopicExtractor:
    """Extract subtopics from PDF using document structure"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
    
    def extract_by_headings(self, 
                           heading_font_threshold: float = None,
                           use_bold: bool = True) -> List[Dict]:
        """
        Extract chunks based on document headings/subtopics.
        
        Args:
            heading_font_threshold: Font size above which text is considered heading
            use_bold: Also detect bold text as headings
        
        Returns:
            List of dicts with 'heading', 'content', 'page', 'level'
        """
        # First pass: detect font sizes and bold text
        all_font_sizes = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():
                            all_font_sizes.append(span["size"])
        
        # Determine heading threshold (e.g., top 20% of font sizes)
        if heading_font_threshold is None:
            all_font_sizes.sort()
            heading_font_threshold = np.percentile(all_font_sizes, 80)
        
        print(f"Heading threshold: {heading_font_threshold:.1f}")
        
        # Second pass: extract content by sections
        sections = []
        current_section = None
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block["type"] != 0:
                    continue
                
                for line in block["lines"]:
                    line_text = ""
                    is_heading = False
                    font_size = 0
                    
                    for span in line["spans"]:
                        text = span["text"].strip()
                        font_size = max(font_size, span["size"])
                        font_name = span["font"].lower()
                        is_bold = "bold" in font_name or "black" in font_name
                        
                        # Check if this is a heading
                        if font_size >= heading_font_threshold or (use_bold and is_bold):
                            is_heading = True
                        
                        line_text += text + " "
                    
                    line_text = line_text.strip()
                    
                    if not line_text:
                        continue
                    
                    # If heading, start new section
                    if is_heading and len(line_text) > 3:  # Avoid single chars
                        # Save previous section
                        if current_section and current_section['content'].strip():
                            sections.append(current_section)
                        
                        # Determine heading level based on font size
                        level = 1 if font_size > heading_font_threshold + 2 else 2
                        
                        current_section = {
                            'heading': line_text,
                            'content': '',
                            'start_page': page_num + 1,
                            'end_page': page_num + 1,
                            'level': level,
                            'font_size': font_size
                        }
                    
                    # Otherwise, add to current section content
                    elif current_section is not None:
                        current_section['content'] += line_text + " "
                        current_section['end_page'] = page_num + 1
        
        # Add final section
        if current_section and current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def extract_by_toc(self) -> List[Dict]:
        """
        Extract chunks using PDF Table of Contents (bookmarks).
        Best method if PDF has proper TOC structure.
        """
        toc = self.doc.get_toc()  # Returns [(level, title, page)]
        
        if not toc:
            print("No TOC found, falling back to heading detection")
            return self.extract_by_headings()
        
        sections = []
        
        for i, (level, title, page_num) in enumerate(toc):
            # Find the end page (start of next section)
            end_page = toc[i + 1][2] - 1 if i + 1 < len(toc) else len(self.doc)
            
            # Extract content between this and next heading
            content = ""
            for p in range(page_num - 1, end_page):
                if p < len(self.doc):
                    page = self.doc[p]
                    content += page.get_text()
            
            # Remove the heading from content if it appears
            content = content.replace(title, "", 1).strip()
            
            sections.append({
                'heading': title,
                'content': content,
                'start_page': page_num,
                'end_page': end_page,
                'level': level
            })
        
        return sections
    
    def extract_hybrid(self) -> List[Dict]:
        """
        Hybrid approach: try TOC first, fall back to heading detection.
        """
        toc = self.doc.get_toc()
        
        if toc and len(toc) > 3:  # Has meaningful TOC
            print("Using TOC-based extraction")
            return self.extract_by_toc()
        else:
            print("Using heading-based extraction")
            return self.extract_by_headings()


class BERTEmbedder:
    """Create embeddings using Sentence-BERT"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize with sentence-transformers model.
        
        Recommended models:
        - 'all-MiniLM-L6-v2': Fast (384 dim)
        - 'all-mpnet-base-v2': Better quality (768 dim)
        - 'multi-qa-mpnet-base-dot-v1': Optimized for Q&A
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def split_long_sections(self, sections: List[Dict], max_tokens: int = 450) -> List[Dict]:
        """
        Split sections that exceed token limit while preserving structure.
        
        Args:
            sections: List of section dicts
            max_tokens: Maximum tokens per chunk
        
        Returns:
            List of sections with long ones split
        """
        processed_sections = []
        
        for section in sections:
            full_text = f"{section['heading']}. {section['content']}"
            tokens = self.tokenizer.tokenize(full_text)
            
            # If section fits, keep as is
            if len(tokens) <= max_tokens:
                processed_sections.append(section)
            else:
                # Split into subsections
                sentences = re.split(r'(?<=[.!?])\s+', section['content'])
                
                sub_chunk = []
                current_tokens = len(self.tokenizer.tokenize(section['heading']))
                part_num = 1
                
                for sentence in sentences:
                    sent_tokens = len(self.tokenizer.tokenize(sentence))
                    
                    if current_tokens + sent_tokens > max_tokens and sub_chunk:
                        # Save current sub-chunk
                        processed_sections.append({
                            'heading': f"{section['heading']} (Part {part_num})",
                            'content': ' '.join(sub_chunk),
                            'start_page': section['start_page'],
                            'end_page': section['end_page'],
                            'level': section['level'],
                            'is_split': True,
                            'part': part_num
                        })
                        
                        sub_chunk = [sentence]
                        current_tokens = len(self.tokenizer.tokenize(section['heading'])) + sent_tokens
                        part_num += 1
                    else:
                        sub_chunk.append(sentence)
                        current_tokens += sent_tokens
                
                # Add final sub-chunk
                if sub_chunk:
                    processed_sections.append({
                        'heading': f"{section['heading']} (Part {part_num})" if part_num > 1 else section['heading'],
                        'content': ' '.join(sub_chunk),
                        'start_page': section['start_page'],
                        'end_page': section['end_page'],
                        'level': section['level'],
                        'is_split': part_num > 1,
                        'part': part_num if part_num > 1 else None
                    })
        
        return processed_sections
    
    def embed_sections(self, sections: List[Dict], batch_size: int = 32) -> Tuple[List[str], np.ndarray]:
        """
        Create embeddings for sections.
        Combines heading + content for better semantic representation.
        
        Returns:
            (texts, embeddings)
        """
        # Combine heading and content
        texts = [f"{s['heading']}. {s['content']}" for s in sections]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return texts, embeddings


class VectorDBManager:
    """Manage ChromaDB storage"""
    
    def __init__(self, collection_name: str = "documents", 
                 chroma_host: str = "localhost", 
                 chroma_port: int = 8000):
        
        self.client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_sections(self, 
                    sections: List[Dict],
                    texts: List[str],
                    embeddings: np.ndarray):
        """Store sections with metadata in ChromaDB"""
        
        ids = [f"section_{i}" for i in range(len(sections))]
        
        # Prepare metadata
        metadata = [
            {
                "heading": s['heading'],
                "start_page": s['start_page'],
                "end_page": s['end_page'],
                "level": s['level'],
                "is_split": s.get('is_split', False)
            }
            for s in sections
        ]
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadata,
            ids=ids
        )
    
    def search(self, query: str, embedder: BERTEmbedder, n_results: int = 5):
        """Search for relevant sections"""
        query_embedding = embedder.model.encode([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        return results


# ============= USAGE EXAMPLE =============

def main():
    pdf_path = "pdfs/llm-book.pdf"
    
    # 1. Extract subtopics/sections
    print("Extracting sections from PDF...")
    extractor = SubtopicExtractor(pdf_path)
    sections = extractor.extract_hybrid()  # Best method
    
    print(f"\nFound {len(sections)} sections:")
    for i, section in enumerate(sections[:5]):  # Show first 5
        print(f"{i+1}. {section['heading']} (Page {section['start_page']}-{section['end_page']})")
        print(f"   Content preview: {section['content'][:100]}...")
    
    # 2. Initialize embedder
    embedder = BERTEmbedder(model_name='all-MiniLM-L6-v2')
    
    # 3. Split long sections if needed
    sections = embedder.split_long_sections(sections, max_tokens=450)
    print(f"\nAfter splitting long sections: {len(sections)} chunks")
    
    # 4. Create embeddings
    print("\nCreating embeddings...")
    texts, embeddings = embedder.embed_sections(sections)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # 5. Store in ChromaDB
    print("\nStoring in ChromaDB...")
    db_manager = VectorDBManager(
        collection_name="llm_book",
        chroma_host="localhost",  # Use "chroma" in Docker
        chroma_port=8000
    )
    
    db_manager.add_sections(sections, texts, embeddings)
    print("✓ Stored successfully!")
    
    # 6. Example search
    query = "What are transformers in NLP?"
    print(f"\n\nSearching for: '{query}'")
    results = db_manager.search(query, embedder, n_results=3)
    
    for i, (doc, meta, dist) in enumerate(zip(results['documents'][0], 
                                               results['metadatas'][0],
                                               results['distances'][0])):
        print(f"\n{i+1}. {meta['heading']} (pages {meta['start_page']}-{meta['end_page']})")
        print(f"   Similarity: {1-dist:.3f}")
        print(f"   {doc[:200]}...")


if __name__ == "__main__":
    main()