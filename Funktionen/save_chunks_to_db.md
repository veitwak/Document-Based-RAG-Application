### Funktion: `save_chunks_to_db()`

**Dateipfad:** `Funktionen/save_chunks_to_db.py`  
**Kategorie:** Datenbank & Persistierung  
**Rolle in der RAG-Architektur:** Speichert verarbeitete Text-Chunks zusammen mit den berechneten Vektor-Embeddings transaktionssicher in PgVector.

---

#### 1. Zweck & Funktionsweise
Nachdem Dokumente eingelesen, in Chunks zerlegt und durch den `EmbeddingClient` vektorisiert wurden, überführt `save_chunks_to_db()` diese Daten in die persistente PostgreSQL-Vektordatenbank.

Die Funktion verknüpft jedes `DocumentChunk`-Objekt mit seinem entsprechenden Vektor-Array (`zip(chunks, embeddings)`) und führt für jeden Datensatz ein parametrisiertes SQL-`INSERT` aus.

#### 2. Schnittstellendefinition

```python
def save_chunks_to_db(
    conn, 
    chunks: List[DocumentChunk], 
    embeddings: List[List[float]]
)
```

* **Parameter:**
  * `conn`: Eine aktive PostgreSQL-Datenbankverbindung.
  * `chunks` (`List[DocumentChunk]`): Liste der DTO-Instanzen mit Text und Metadaten.
  * `embeddings` (`List[List[float]]`): Liste der zugehörigen 384-dimensionalen Vektoren.
* **Rückgabewert:** Keiner (`None`). Gibt Statusmeldung auf der Konsole aus.

#### 3. Detaillierter Ablauf
1. **Cursor-Kontext:** Öffnet einen Transaktions-Cursor (`with conn.cursor() as cur:`).
2. **Synchrones Iterieren:** Mittels `zip(chunks, embeddings)` werden Textfragment, Metadaten und Embedding paarweise durchlaufen.
3. **Parametrisiertes SQL-Kommando:**
   ```sql
   INSERT INTO document_chunks (file_name, page_number, document_type, content, embedding)
   VALUES (%s, %s, %s, %s, %s);
   ```
4. **Commit:** `conn.commit()` schließt die Transaktion atomar ab (ACID-Prinzip).

#### 4. Wichtige Architektur- & Design-Entscheidungen
* **SQL-Injection Schutz:**  
  Durch die konsequente Verwendung von parametrisierten Abfragen (`%s`) ist die Datenbank vollständig vor SQL-Injections geschützt, selbst wenn die eingelesenen Bewerbungsdokumente bösartigen SQL-Code enthalten sollten ("Prompt/Data Injection via Resume").
* **Transaktionale Integrität:**  
  Tritt während des Schreibens ein Fehler auf, wird die Transaktion zurückgerollt, sodass keine unvollständigen oder inkonsistenten Datenbestände verbleiben.

#### 5. Häufige Fragen & Design-Entscheidungen (FAQ)

> **F: Wie skaliert diese Funktion bei Millionen von Chunks?**  
> *Antwort:* Für sehr große Datenmengen (Big Data Ingestion) würde man von sequenziellen `cur.execute`-Aufrufen auf Bulk-Operationen wie `psycopg2.extras.execute_values` oder den PostgreSQL-Befehl `COPY` umstellen. Dadurch reduziert sich der Netzwerk-Roundtrip zwischen Anwendung und Datenbank massiv.

---

#### 6. Vollständiger Quellcode

```python
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
```
