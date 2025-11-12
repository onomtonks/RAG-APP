# src/pdf_reader.py
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

def extract_text_and_toc(pdf_path, max_pages=None):
    """
    Extract:
    - full text
    - text per page (for TOC mapping)
    - table of contents (outline)
    """
    reader = PdfReader(pdf_path)
    full_text = ""
    page_texts = []

    # ✅ Extract TOC
    toc = []
    try:
        outlines = reader.outline
        for item in outlines:
            if hasattr(item, "title") and hasattr(item, "page"):
                toc.append({
                    "title": item.title,
                    "page": reader.get_destination_page_number(item)
                })
    except Exception:
        pass

    # ✅ Extract text page by page (with OCR fallback)
    for i, page in enumerate(reader.pages):
        if max_pages and i >= max_pages:
            break

        page_text = page.extract_text() or ""

        # OCR fallback for scanned PDFs
        try:
            images = convert_from_path(pdf_path, dpi=200, first_page=i+1, last_page=i+1)
            for img in images:
                page_text += pytesseract.image_to_string(img)
        except Exception:
            pass

        page_texts.append(page_text)
        full_text += page_text + "\n"

    return full_text, page_texts, toc


def segment_text_by_toc(page_texts, toc):
    """
    Divide text into sections based on Table of Contents.
    Each section includes:
    - title
    - start_page
    - text
    """
    if not toc:
        # fallback: entire text is one section
        return [{"title": "Full Document", "start_page": 1, "text": "\n".join(page_texts)}]

    sections = []
    toc_sorted = sorted(toc, key=lambda x: x["page"])
    num_pages = len(page_texts)

    for i, entry in enumerate(toc_sorted):
        start_page = entry["page"]
        end_page = toc_sorted[i + 1]["page"] if i + 1 < len(toc_sorted) else num_pages
        section_text = "\n".join(page_texts[start_page:end_page])
        sections.append({
            "title": entry["title"],
            "start_page": start_page,
            "text": section_text
        })

    return sections
