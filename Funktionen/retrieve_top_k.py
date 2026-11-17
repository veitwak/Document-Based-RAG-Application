from typing import List, Tuple

def retrieve_top_k(conn, query_embedding: List[float], top_k: int = 10) -> List[dict]:
    """Führt Similarity Search über Kosinusdistanz (<=>) aus und ermittelt Top-K Chunks."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT file_name, page_number, document_type, content, 
                   1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s;
        """, (query_embedding, query_embedding, top_k))
        rows = cur.fetchall()

        return [
            {
                "file_name": r[0],
                "page_number": r[1],
                "document_type": r[2],
                "content": r[3],
                "similarity": float(r[4])
            }
            for r in rows
        ]
