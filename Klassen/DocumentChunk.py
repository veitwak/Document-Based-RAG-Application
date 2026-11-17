from dataclasses import dataclass

@dataclass
class DocumentChunk:
    text: str
    file_name: str
    page_number: int
    document_type: str