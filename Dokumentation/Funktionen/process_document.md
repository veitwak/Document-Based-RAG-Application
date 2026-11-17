### Funktion: `process_document()`

**Dateipfad:** `Funktionen/process_document.py`  
**Kategorie:** Ingestion & Orchestrierung  
**Rolle in der RAG-Architektur:** Koordinierende Schnittstelle der Ingestion-Pipeline für eine einzelne Datei. Verknüpft Textextraktion mit seitenweisem Chunking und instanziiert `DocumentChunk`-Objekte.

---

#### 1. Zweck & Funktionsweise
Die Funktion `process_document()` bildet die Brücke zwischen dem Dateisystem und dem Vektorspeicher. Sie kapselt die vollständige Vorverarbeitung einer Datei:
1. Ermitteln der Dateimetadaten (`file_name`, `document_type`).
2. Aufruf von `extract_pages_from_file()` zur seitenweisen Textextraktion (inklusive OCR-Fallback).
3. Seitenweises Aufrufen von `recursive_chunk_text()`, damit Chunks nicht seitenübergreifend verschmelzen und die Seitenzahl exakt erhalten bleibt.
4. Zusammenführen der Segmente in typisierte `DocumentChunk`-Objekte.

#### 2. Schnittstellendefinition

```python
def process_document(file_path: str) -> List[DocumentChunk]
```

* **Parameter:**
  * `file_path` (`str`): Dateipfad zur verarbeitenden Datei (z. B. `Dateien/Lebenslauf.pdf`).
* **Rückgabewert:**
  * `List[DocumentChunk]`: Eine Liste vollständig angereicherter `DocumentChunk`-Instanzen, bereit für Vektorisierung und Persistierung.

#### 3. Detaillierter Ablauf
1. **Metadaten-Extraktion:**  
   `os.path.basename` liefert den sauberen Dateinamen (z. B. `Lebenslauf.pdf`). `os.path.splitext` liefert das Format (`pdf`, `txt`).
2. **Seitenweises Einlesen:**  
   `pages = extract_pages_from_file(file_path)` liefert eine Liste von `(page_num, content)`-Tupeln.
3. **Chunking pro Seite:**  
   Für jede Seite wird geprüft, ob Text existiert (`if not content: continue`).  
   Anschließend wird `recursive_chunk_text(content, chunk_size=500, overlap=50)` ausgeführt.
4. **DTO-Instanziierung:**  
   Jedes erzeugte Textfragment wird zusammen mit `file_name`, `page_number` und `document_type` als `DocumentChunk` in die Ergebnisliste eingetragen.

#### 4. Wichtige Architektur- & Design-Entscheidungen
* **Seiten-Isoliertes Chunking:**  
  Ein Chunk erstreckt sich niemals über Seitengrenzen hinweg. Dadurch ist die Zuordnung im Zitat eindeutig ("Gefunden auf Seite 2").
* **Single Responsibility Principle:**  
  Die Funktion enthält weder Low-Level-PDF-Handling noch Vektorisierungslogik, sondern verbindet spezialisierte Module.

#### 5. Häufige Fragen & Design-Entscheidungen (FAQ)

> **F: Warum chunking Sie seitenweise und nicht den gesamten Dokumententext auf einmal?**  
> *Antwort:* Erstens für die exakte Zitation (jede Information bleibt einer Seitenzahl zugeordnet). Zweitens weisen Seitenumbrüche in Lebensläufen und Zeugnissen oft natürliche thematische Schnitte auf (z. B. Seite 1 CV, Seite 2 Anlagen), die durch Dokumenten-weites Zusammenfügen semantisch vermischt würden.

---

#### 6. Vollständiger Quellcode

```python
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
```
