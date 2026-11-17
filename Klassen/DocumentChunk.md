### Klasse: `DocumentChunk`

**Dateipfad:** `Klassen/DocumentChunk.py`  
**Kategorie:** Datenmodell / Data-Transfer-Object (DTO)  
**Rolle in der RAG-Architektur:** Repräsentiert ein semantisch isoliertes Textsegment im Arbeitsspeicher (In-Memory) inklusive aller Provenienz-Metadaten zur lückenlosen Nachvollziehbarkeit und Quellenangabe.

---

#### 1. Zweck & Funktionsweise
In der Ingestion-Pipeline werden Dokumente (PDF, TXT) nicht als amorpher Textblock verarbeitet, sondern in handhabbare Einheiten zerlegt. Die Klasse `DocumentChunk` kapselt diese Fragmente als strukturiertes Datenobjekt. 

Anstatt temporäre `.txt`-Zwischendateien auf der Festplatte anzulegen (was I/O-Overhead erzeugt und Datenschutzrisiken birgt), verbleiben die Chunks bis zur Speicherung in PostgreSQL/PgVector rein im Arbeitsspeicher (`In-Memory Representation`).

#### 2. Attribute & Datentypen

| Attribut | Typ | Beschreibung |
| :--- | :--- | :--- |
| `text` | `str` | Der extrahierte und segmentierte Textabschnitt (Richtwert: max. ca. 500 Zeichen). |
| `file_name` | `str` | Der Dateiname der Ursprungsquelle (z. B. `Lebenslauf.pdf`). Ermöglicht Quellenzitierung. |
| `page_number` | `int` | Die 1-basierte Seitenzahl innerhalb des Quelldokuments, von der das Segment stammt. |
| `document_type` | `str` | Dateiformat bzw. Extension ohne Punkt (z. B. `pdf`, `txt`) für spätere Filterungen. |

#### 3. Wichtige Architektur- & Design-Entscheidungen
* **Verwendung von `@dataclass`:**  
  Python Dataclasses reduzieren Boilerplate-Code drastisch, erzeugen automatisch typsichere `__init__`, `__repr__` und `__eq__` Methoden und bieten hohe Performance bei geringem Speicher-Footprint.
* **Explainable AI & Provenienz:**  
  Durch die Verknüpfung jedes Textfragments mit `file_name` und `page_number` kann das System in der Retrieval-Phase dem HR-Benutzer exakt nachweisen, auf welcher Seite welcher Datei eine Information gefunden wurde (Audit-Trail).

#### 4. Häufige Fragen & Design-Entscheidungen (FAQ)

> **F: Warum speichern Sie Seitenzahlen und Dateinamen direkt an jedem Chunk?**  
> *Antwort:* Im HR- und Enterprise-Kontext ist reine Faktenausgabe nicht ausreichend. Antworten müssen überprüfbar sein. Durch die Metadaten kann der HR-Prüfer bei Rückfragen sofort die entsprechende Originalseite im PDF nachschlagen.

> **F: Warum eine `dataclass` und kein einfaches Python-Dictionary?**  
> *Antwort:* Eine Dataclass erzwingt eine formale Schnittstelle mit Typannotationen. Das verhindert Tippfehler bei Schlüsselnamen, ermöglicht statische Typprüfung (z. B. mit `mypy`) und erleichtert das Refactoring in größeren Codebasen.

---

#### 5. Vollständiger Quellcode

```python
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    text: str
    file_name: str
    page_number: int
    document_type: str
```
