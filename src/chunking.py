#from unstructured.partition.pdf import partition_pdf
#from unstructured.documents.elements import Table, CompositeElement

def chunks(file_path):
    """
    Split a PDF into chunks: text and tables.
    """
    pdf_chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        hi_res_model_name="detectron2_onnx",
        extract_image_block_to_payload=True,

        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    tables = []
    texts = []
    
    for chunk in pdf_chunks:
        if isinstance(chunk, Table):
            tables.append(chunk)

        if isinstance(chunk, CompositeElement):
            texts.append(chunk)

    return tables, texts


if __name__ == "__main__":
    from unstructured.partition.pdf import partition_pdf
    from unstructured.documents.elements import Table
    
    import nltk

    nltk.download("punkt_tab")

    pdf_chunks = partition_pdf(
        filename='pdfs/llm-book.pdf',
        strategy="hi_res",
        hi_res_model_name="detectron2_onnx",
        extract_image_block_to_payload=True,
        ocr_languages=["eng"],  # enable OCR
    )

    tables = []
    texts = []

    for chunk in pdf_chunks:
        if isinstance(chunk, Table):
            tables.append(chunk)
        elif isinstance(chunk, TextBlock):
            texts.append(chunk)

    print("Tables detected:", len(tables))
    print("Text blocks detected:", len(texts))
    if texts:
        print("Sample text:", texts[0].text[:200])
