### Funktion: `retrieve_top_k()`

**Dateipfad:** `Funktionen/retrieve_top_k.py`  
**Kategorie:** Retrieval & Vektorsuche  
**Rolle in der RAG-Architektur:** Führt eine Ähnlichkeitssuche (Similarity Search) über Kosinusdistanz im hochdimensionalen Vektorraum aus und selektiert die $K$ relevantesten Text-Chunks.

---

#### 1. Zweck & Funktionsweise
In der Retrieval-Phase wandelt das System die Benutzerfrage in einen Suchvektor um. `retrieve_top_k()` übergibt diesen Vektor an PostgreSQL/PgVector und nutzt den vorberechneten HNSW-Index, um die semantisch am nächsten liegenden Textfragmente blitzschnell aufzufinden.

Die mathematische Grundlage bildet die **Kosinus-Distanz** (Cosine Distance). In PgVector wird diese durch den Operator `<=>` repräsentiert:
$$\text{Cosine Distance } D_{\cos}(u, v) = 1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

Da eine kleinere Distanz eine höhere Ähnlichkeit bedeutet, sortiert die Abfrage aufsteigend nach Distanz (`ORDER BY embedding <=> %s::vector ASC`) und wandelt den Wert in einen Ähnlichkeitsscore um:
$$\text{Similarity } = 1 - D_{\cos}(u, v)$$

#### 2. Schnittstellendefinition

```python
def retrieve_top_k(
    conn, 
    query_embedding: List[float], 
    top_k: int = 10
) -> List[dict]
```

* **Parameter:**
  * `conn`: Eine aktive PostgreSQL-Datenbankverbindung.
  * `query_embedding` (`List[float]`): Der 384-dimensionale Vektor der Benutzeranfrage.
  * `top_k` (`int`, Standard `10`): Maximale Anzahl an zurückzugebenden Treffern.
* **Rückgabewert:**
  * `List[dict]`: Liste von Dictionaries, sortiert nach absteigender Relevanz:
    ```python
    {
        "file_name": str,       # Ursprungsdatei
        "page_number": int,     # Seitennummer
        "document_type": str,   # Dateiendung
        "content": str,         # Extrahierter Textabschnitt
        "similarity": float     # Ähnlichkeitsscore zwischen 0.0 und 1.0
    }
    ```

#### 3. Wichtige Architektur- & Design-Entscheidungen
* **Index-gestützte Beschleunigung (HNSW):**  
  Durch die direkte Verwendung von `<=>` in der `ORDER BY`-Klausel erkennt der PostgreSQL-Query-Planner den HNSW-Index (`idx_document_chunks_hnsw`). Anstatt jeden Vektor der Tabelle sequenziell zu vergleichen (Full Table Scan in $O(N)$), navigiert die Suche den hierarchischen Graph in logarithmischer Zeit $O(\log N)$.
* **Score-Normalisierung:**  
  Die Berechnung `1 - (embedding <=> %s::vector)` liefert einen intuitiv interpretierbaren Ähnlichkeitsscore, der für Schwellenwert-Filter (Thresholding) oder UI-Ausgaben verwendet werden kann.

#### 4. Häufige Fragen & Design-Entscheidungen (FAQ)

> **F: Warum nutzen Sie Kosinus-Distanz (`<=>`) statt Euklidischer Distanz (`<->` / L2)?**  
> *Antwort:* Die euklidische Distanz misst den absoluten Abstand im Raum und ist stark von der Textlänge bzw. Vektornorm abhängig. Die Kosinus-Distanz misst ausschließlich den Winkel zwischen den Richtungsvektoren. Zwei Texte mit denselben Schlüsselbegriffen, aber unterschiedlicher Länge zeigen daher nahezu denselben Kosinuswinkel.

> **F: Wie wählen Sie den Parameter `top_k` in der Praxis?**  
> *Antwort:* Ein zu kleines $K$ (z. B. $K=1$) birgt das Risiko, entscheidende Nebensätze oder über mehrere Seiten verteilte Details zu verpassen. Ein zu großes $K$ ($K > 50$) füllt das Kontextfenster mit Rauschen ("Lost in the Middle"-Phänomen) und erhöht Latenz und Kosten. $K = 5$ bis $K = 20$ ist der empirisch bewährte Sweet-Spot.

---

#### 5. Vollständiger Quellcode

```python
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
```
