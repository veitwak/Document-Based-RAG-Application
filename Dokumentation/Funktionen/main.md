### Einstiegspunkt: `main.py` / Funktion `main()`

**Dateipfad:** `main.py`  
**Kategorie:** System-Orchestrierung & CLI  
**Rolle in der RAG-Architektur:** Zentraler Haupteinstiegspunkt (Main Orchestrator), der Initialisierung, Ingestion-Pipeline und interaktive Frage-Antwort-Schleife zusammenführt.

---

#### 1. Zweck & Funktionsweise
Die Funktion `main()` orchestriert den gesamten Lebenszyklus der RAG-Anwendung:
1. **Infrastruktur-Check:** Stellt die Verbindung zur Vektordatenbank her. Schlägt diese fehl, erhält der Entwickler eine klare Anweisung (`docker compose up -d`).
2. **Setup:** Initialisiert den `EmbeddingClient` und ruft `init_database()` auf, um Tabellen und HNSW-Indizes passend zur Vektordimension zu synchronisieren.
3. **Flexible Dateiaufnahme:** Verarbeitet Dokumentenpfade entweder über CLI-Argumente (`python main.py datei1.pdf,datei2.txt`) oder interaktiv über ein Konsolen-Prompt.
4. **Batch Ingestion & Indexing:** Verarbeitet alle angegebenen Dateien via `process_document()`, berechnet alle Embeddings gebündelt im Batch und persistiert sie über `save_chunks_to_db()`.
5. **Interaktive REPL (Read-Eval-Print-Loop):** Startet eine Dauerschleife, in der der Nutzer beliebig viele Fragen an den HR-Assistenten stellen kann, bis er mit Enter oder `quit`/`exit` abbricht.

#### 2. Schnittstellendefinition

```python
def main()
```

* **Aufruf:** `python main.py [dateipfade...]`
* **Rückgabewert:** Keiner (`None`). Beendet den Prozess bei DB-Fehlern via `sys.exit(1)`.

#### 3. Detaillierter Ablauf
1. **DB-Verbindung:** `conn = get_db_connection()`.
2. **Embedding & Schema:** `embed_client = EmbeddingClient()`, danach `init_database(conn, embed_client.dimension)`.
3. **Dateipfade parsen:**  
   Liest `sys.argv[1:]`. Falls leer, fragt ein `input()`-Prompt nach Pfaden. Unterstützt Trennung durch Kommas oder Leerzeichen sowie Quotes (`"`, `'`).
4. **Dokumentenverarbeitung:**  
   Iteriert über alle Pfade. Ist die Datei vorhanden, erzeugt `process_document(path)` die Liste `chunks`.
5. **Batch-Vektorisierung & Persistierung:**  
   Falls Chunks vorhanden sind:
   ```python
   texts = [c.text for c in all_chunks]
   embeddings = embed_client.embed_texts(texts)
   save_chunks_to_db(conn, all_chunks, embeddings)
   ```
6. **Interaktiver Frage-Modus:**  
   Dauerschleife mit `user_query = input("\nDeine Frage: ")`. Ruft für jede Frage `ask_hr_assistant(conn, embed_client, user_query, top_k=20)` auf.
7. **Clean Shutdown:** Schließt die DB-Verbindung sauber via `conn.close()`.

#### 4. Wichtige Architektur- & Design-Entscheidungen
* **Batch-Encoding vor dem DB-Write:**  
  Texte werden zuerst komplett gesammelt und als Liste an `embed_client.embed_texts(texts)` übergeben. Dies maximiert die Auslastung der CPU/Vektorbeschleunigung gegenüber Einzelaufrufen.
* **Separation of Concerns:**  
  `main.py` enthält keine SQL-Statements oder PDF-Parsing-Details, sondern delegiert alle Aufgaben an modulare Einzelkomponenten.

#### 5. Häufige Fragen & Design-Entscheidungen (FAQ)

> **F: Warum haben Sie in `main()` `top_k=20` an `ask_hr_assistant` übergeben?**  
> *Antwort:* Lebensläufe und Zeugnisse sind oft stichpunktartig aufgebaut. Durch $K=20$ werden auch weiter verstreute Fähigkeiten, Zertifikate und Zwischenstationen vollständig erfasst, ohne dass das Kontextfenster von Llama 3.1 (8k Kontext) überfordert wird.

> **F: Wie könnte man aus der CLI-Anwendung eine REST-API machen?**  
> *Antwort:* Man würde die Initialisierungslogik aus `main()` in das Lifespan-Event von FastAPI auslagern und zwei Endpunkte definieren: `POST /upload` (ruft `process_document`, `embed_texts`, `save_chunks_to_db` auf) und `POST /query` (ruft `ask_hr_assistant` auf).

---

#### 6. Vollständiger Quellcode

```python
# import bibs

import os
import sys
from typing import List

# import Klassen
from Klassen.EmbeddingClient import EmbeddingClient
from Klassen.DocumentChunk import DocumentChunk

# import Funktionen

from Funktionen.process_document import process_document
from Funktionen.get_db_connection import get_db_connection
from Funktionen.save_chunks_to_db import save_chunks_to_db
from Funktionen.init_database import init_database
from Funktionen.ask_hr_assistant import ask_hr_assistant


def main():
    # Datenbankverbindung herstellen
    try:
        conn = get_db_connection()
        print("Mit PostgreSQL verbunden.")
    except Exception as e:
        print(f"[FEHLER]: {e}")
        print("Hinweis: Bitte starte den Datenbank-Container mit 'docker compose up -d'.")
        sys.exit(1)

    # Embedding-Client initialisieren
    embed_client = EmbeddingClient()
    init_database(conn, embed_client.dimension)

    # Ingestion & Chunking
    raw_files = sys.argv[1:]
    if not raw_files:
        entered = input("Dateipfad(e) eingeben (kommagetrennt, z. B. Dateien/Lebenslauf.pdf): ").strip()
        if entered:
            raw_files = [entered]

    # Unterstützt CLI-Argumente (Leerzeichen) und Kommaschreibweise
    input_files = []
    for arg in raw_files:
        input_files.extend([p.strip().strip("'\"") for p in arg.split(",") if p.strip()])

    all_chunks: List[DocumentChunk] = []

    for path in input_files:
        if os.path.exists(path):
            print(f"[Ingestion] Verarbeite Datei: {path}")
            chunks = process_document(path)
            all_chunks.extend(chunks)
        else:
            print(f"[Ingestion] Datei nicht gefunden: {path}")

    if all_chunks:
        # Embeddings berechnen und in PgVector speichern
        texts = [c.text for c in all_chunks]
        embeddings = embed_client.embed_texts(texts)
        save_chunks_to_db(conn, all_chunks, embeddings)

    # Interaktiver Modus
    print("\n")
    print("Interaktiver Modus: Stelle eigene Fragen (oder Enter zum Beenden)")
    while True:
        user_query = input("\nDeine Frage: ").strip()
        if not user_query or user_query.lower() in ("exit", "quit", "q"):
            break
        ask_hr_assistant(conn, embed_client, user_query, top_k=20)

    conn.close()


if __name__ == "__main__":
    main()
```
