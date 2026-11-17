from pgvector.psycopg2 import register_vector

def init_database(conn, dimension: int):
    """Initialisiert pgvector-Erweiterung, Tabelle und HNSW-Index."""
    # 1. Erst Extension in der DB aktivieren
    with conn.cursor() as cur:
        # DB-Tabelle leeren, damit keine Duplikate entstehen
        cur.execute("TRUNCATE TABLE document_chunks;")
    conn.commit()

    # 2. Erst jetzt pgvector für psycopg2 registrieren
    register_vector(conn)
    
    with conn.cursor() as cur:
        # Prüfung, ob die Tabelle bereits mit anderer Dimension existiert
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'document_chunks' AND column_name = 'embedding';
        """)
        if cur.fetchone():
            cur.execute("""
                SELECT atttypmod 
                FROM pg_attribute 
                WHERE attrelid = 'document_chunks'::regclass AND attname = 'embedding';
            """)
            col_info = cur.fetchone()
            if col_info and col_info[0] != dimension and col_info[0] > 0:
                print(f"[DB] Dimension geändert ({col_info[0]} -> {dimension}). Erstelle Tabelle neu...")
                cur.execute("DROP TABLE document_chunks CASCADE;")

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                file_name VARCHAR(255) NOT NULL,
                page_number INTEGER NOT NULL,
                document_type VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                embedding vector({dimension}) NOT NULL
            );
        """)

        # HNSW-Index mit Kosinusähnlichkeit (vector_cosine_ops)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_hnsw 
            ON document_chunks USING hnsw (embedding vector_cosine_ops);
        """)
    conn.commit()

