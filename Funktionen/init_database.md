### Funktion: `init_database()`

**Dateipfad:** `Funktionen/init_database.py`  
**Kategorie:** Datenbank & Schema-Management  
**Rolle in der RAG-Architektur:** Initialisiert die PgVector-Erweiterung, stellt das Tabellenschema sicher, führt automatisierte Dimensions-Migrationen durch und erzeugt den HNSW-Vektorindex.

---

#### 1. Zweck & Funktionsweise
Die Vektorsuche erfordert in PostgreSQL eine spezialisierte Tabellenstruktur mit dem Datentyp `vector(dim)` und einen Approximate Nearest Neighbor (ANN) Index. 

`init_database()` automatisiert das gesamte Setup:
1. Bereinigt vorhandene Datensätze (`TRUNCATE TABLE`), um Duplikate bei erneuten Durchläufen zu verhindern.
2. Registriert den PgVector-Datentyp für `psycopg2` via `register_vector(conn)`.
3. **Dimensions-Konsistenzprüfung:** Wechselt man das Embedding-Modell (z. B. von 1536 auf 384 Dimensionen), würde PostgreSQL mit einem Typfehler abstürzen. Die Funktion prüft über den PostgreSQL-Systemkatalog (`pg_attribute`), ob die vorhandene Spaltendimension noch zur Modell-Dimension passt. Bei einer Diskrepanz wird die Tabelle automatisch gedroppt und mit der neuen Dimension neu erzeugt.
4. Erstellt die Tabelle `document_chunks` mit Metadaten-Spalten.
5. Erzeugt einen **HNSW-Index** (Hierarchical Navigable Small World) mit Kosinus-Distanz (`vector_cosine_ops`).

#### 2. Schnittstellendefinition

```python
def init_database(conn, dimension: int)
```

* **Parameter:**
  * `conn`: Eine aktive PostgreSQL-Verbindung.
  * `dimension` (`int`): Die Vektordimension des verwendeten Embedding-Modells (z. B. `384`).
* **Rückgabewert:** Keiner (`None`).

#### 3. Detaillierter Ablauf
1. `TRUNCATE TABLE document_chunks;`: Setzt die Tabelle zurück.
2. `register_vector(conn)`: Ermöglicht es `psycopg2`, Python-Listen direkt als PgVector-Vektoren zu serialisieren/deserialisieren.
3. **Systemkatalog-Inspektion:**
   * Fragt `information_schema.columns` ab, ob die Spalte `embedding` existiert.
   * Liest `atttypmod` aus `pg_attribute` aus (enthält die konfigurierte Vektorlänge).
   * Weicht die Dimension ab (`col_info[0] != dimension`), wird `DROP TABLE document_chunks CASCADE;` ausgeführt.
4. **Tabellen-DDL:**
   * `id SERIAL PRIMARY KEY`
   * `file_name VARCHAR(255) NOT NULL`
   * `page_number INTEGER NOT NULL`
   * `document_type VARCHAR(50) NOT NULL`
   * `content TEXT NOT NULL`
   * `embedding vector(dimension) NOT NULL`
5. **HNSW Index-DDL:**
   * `CREATE INDEX IF NOT EXISTS idx_document_chunks_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops);`

#### 4. Wichtige Architektur- & Design-Entscheidungen
* **Warum HNSW statt IVFFlat?**  
  * *HNSW (Hierarchical Navigable Small World):* Bietet extrem hohen Recall (> 95-98 %), logarithmische Suchkomplexität $O(\log N)$ und erfordert kein vorzeitiges "Training" auf Beispieldaten. Es funktioniert ab dem ersten eingefügten Vektor sofort performant.
  * *IVFFlat (Inverted File Flat):* Benötigt zwingend eine Trainingsphase und clusterbasierte Zentrenberechnung, was bei dynamisch wachsenden Dokumentenbeständen zu Qualitätsverlusten führt.
* **Kosinus-Metrik (`vector_cosine_ops`):**  
  Optimiert für normalisierte Embeddings aus Transformer-Modellen.

#### 5. Häufige Fragen & Design-Entscheidungen (FAQ)

> **F: Warum ist die dynamische Prüfung der Dimension über `pg_attribute` wichtig?**  
> *Antwort:* In der Prototypen- und Evaluierungsphase werden häufig verschiedene Embedding-Modelle getestet (z. B. OpenAI mit 1536 vs. lokales MiniLM mit 384 Dimensionen). Ohne diese Prüfung schlägt jeder `INSERT` mit einem kryptischen Datenbankfehler fehl, wenn die Tabelle noch mit der alten Dimension persistiert ist.

---

#### 6. Vollständiger Quellcode

```python
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
```
