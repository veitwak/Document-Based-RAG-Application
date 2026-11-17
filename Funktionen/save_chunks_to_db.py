from typing import List, Tuple
from Klassen.DocumentChunk import DocumentChunk

def save_chunks_to_db(conn, chunks: List[DocumentChunk], embeddings: List[List[float]]):
    """Speichert Chunks inklusive Embeddings in die Vektordatenbank."""
    with conn.cursor() as cur:
        for chunk, emb in zip(chunks, embeddings):
            cur.execute("""
                INSERT INTO document_chunks (file_name, page_number, document_type, content, embedding)
                VALUES (%s, %s, %s, %s, %s);
            """, (chunk.file_name, chunk.page_number, chunk.document_type, chunk.text, emb))
    conn.commit()
    print(f"[DB] {len(chunks)} Chunks erfolgreich in PgVector indiziert.")
