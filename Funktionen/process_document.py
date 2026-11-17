import os
from typing import List, Tuple
from Klassen.DocumentChunk import DocumentChunk
from Funktionen.extract_pages_from_file import extract_pages_from_file
from Funktionen.recursive_chunk_text import recursive_chunk_text

def process_document(file_path: str) -> List[DocumentChunk]:
    """Liest ein Dokument ein und erzeugt semantische In-Memory Chunks."""
    file_name = os.path.basename(file_path)
    doc_type = os.path.splitext(file_path)[1].lower().replace(".", "")

    pages = extract_pages_from_file(file_path)
    all_chunks = []

    for page_num, content in pages:
        if not content:
            continue
        text_chunks = recursive_chunk_text(content, chunk_size=500, overlap=50)
        for chunk in text_chunks:
            all_chunks.append(DocumentChunk(
                text=chunk,
                file_name=file_name,
                page_number=page_num,
                document_type=doc_type
            ))

    return all_chunks
