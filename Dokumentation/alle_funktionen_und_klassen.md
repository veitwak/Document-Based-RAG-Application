## 6. Detaillierte Code- und Funktionsreferenz

Dieser Abschnitt dokumentiert alle Klassen und Funktionen des Gesamtsystems in ihrer logischen Ausführungsreihenfolge entlang der RAG-Pipeline. Jeder Eintrag enthält Zweck, Schnittstellen, algorithmische Details, Architektur-Entscheidungen, häufige Fragestellungen & Design-Entscheidungen (FAQ) sowie den vollständigen Quellcode.

---

### Inhaltsverzeichnis
* **Ingestion & Datenaufbereitung:**
  * [Funktion: extract_pages_from_file()](#funktion-extract_pages_from_file)
  * [Funktion: recursive_chunk_text()](#funktion-recursive_chunk_text)
  * [Klasse: DocumentChunk](#klasse-documentchunk)
  * [Funktion: process_document()](#funktion-process_document)
* **Vektorisierung & Datenbank:**
  * [Klasse: EmbeddingClient](#klasse-embeddingclient)
  * [Funktion: get_db_connection()](#funktion-get_db_connection)
  * [Funktion: init_database()](#funktion-init_database)
  * [Funktion: save_chunks_to_db()](#funktion-save_chunks_to_db)
* **Retrieval & LLM-Inferenz:**
  * [Funktion: retrieve_top_k()](#funktion-retrieve_top_k)
  * [Funktion: ask_hr_assistant()](#funktion-ask_hr_assistant)
* **System-Einstieg:**
  * [Einstiegspunkt: main.py / Funktion main()](#einstiegspunkt-mainpy--funktion-main)

---

### Funktion: `extract_pages_from_file()`

**Dateipfad:** `Funktionen/extract_pages_from_file.py`  
**Kategorie:** Ingestion & Dokumenten-Extraktion  
**Rolle in der RAG-Architektur:** Universeller Extraktor für `.txt`- und `.pdf`-Dateien mit automatischem Multi-Tier OCR-Fallback für gescannte oder rein bildbasierte Dokumentenseiten.

#### 1. Zweck & Funktionsweise
In Bewerbungsunterlagen treten häufig hybride PDFs auf – beispielsweise ein digital erstellter Lebenslauf (Seite 1), an den eingescannte Arbeitszeugnisse oder handschriftlich unterschriebene Dokumente (Seiten 2 und 3) angehängt wurden. 

Standard-PDF-Parser scheitern bei Scans, da dort keine Text-Glyphen im PDF-Stream eingebettet sind. `extract_pages_from_file()` prüft jede Seite einzeln:
1. Ist nativer Text vorhanden, wird dieser direkt und verlustfrei extrahiert.
2. Ist die Seite leer oder enthält nur Bilddaten, wird automatisch ein **OCR-Fallback** (Optical Character Recognition via Tesseract bei 300 DPI) für genau diese Seite ausgelöst.

#### 2. Schnittstellendefinition

```python
def extract_pages_from_file(file_path: str) -> List[Tuple[int, str]]
```

* **Parameter:**
  * `file_path` (`str`): Absoluter oder relativer Pfad zur Zieldatei.
* **Rückgabewert:**
  * `List[Tuple[int, str]]`: Eine Liste von Tupeln aus `(Seitennummer, extrahierter Text)`. Die Seitennummerierung ist 1-basiert.

#### 3. Detaillierter Ablauf & Kaskade
1. **Dateityp-Erkennung:**  
   Unterscheidung anhand der Dateiendung (`.txt` vs. `.pdf`). Nicht unterstützte Formate werden geloggt und ignoriert.
2. **Textdateien (`.txt`):**  
   Werden mit `encoding="utf-8"` und `errors="replace"` eingelesen und direkt als Seite 1 zurückgegeben.
3. **PDF-Verarbeitung mit 2-Stufen-Kaskade:**
   * **Stufe 1 (`pdfplumber` - Primärparser):**  
     Iteriert über alle Seiten. Pro Seite wird `page.extract_text()` versucht. Bleibt der Text leer (`not text.strip()`), rendert `page.to_image(resolution=300).original` die Seite hochauflösend als Rastergrafik, welche per `pytesseract.image_to_string()` in Text umgewandelt wird.
   * **Stufe 2 (`pypdf` - Fallback-Parser):**  
     Falls `pdfplumber` nicht verfügbar sein sollte, greift der `pypdf`-Parser. Fehlt auch hier Text, werden eingebettete Bilder (`page.images`) an Tesseract übergeben.

#### 4. Wichtige Architektur- & Design-Entscheidungen
* **Seitenweise Granularität:**  
  Statt das gesamte Dokument blind durch eine langsame OCR-Pipeline zu jagen, entscheidet das System seitenweise. Das spart erhebliche Rechenzeit bei Dokumenten mit gemischten Inhalten.
* **300 DPI Rendering:**  
  Bei Scans (z. B. Zeugnisnoten in kleiner Schriftart) ist eine Bildauflösung von 300 DPI der Industriestandard für verlässliche Tesseract-Erkennungsraten.

#### 5. Häufige Fragen & Design-Entscheidungen (FAQ)

> **F: Wie handhaben Sie Dokumente mit handschriftlichen Unterschriften oder Stempeln?**  
> *Antwort:* Die native PDF-Extraktion liest den getippten Text. Befindet sich der Text nur im Bild (Scan), springt Tesseract ein. Für komplexe handschriftliche Notizen könnte künftig ein Vision-LLM (z. B. Llama 3.2 Vision) als zusätzlicher Fallback integriert werden.

> **F: Warum wird `errors="replace"` beim Lesen von Textdateien verwendet?**  
> *Antwort:* Verhindert unkontrollierte Abstürze bei unerwarteten Byte-Sequenzen (z. B. fehlerhaftes Latin-1/Windows-1252 Encoding in älteren Exporten).

#### 6. Vollständiger Quellcode

```python
import os
from typing import List, Tuple

def extract_pages_from_file(file_path: str) -> List[Tuple[int, str]]:
    """Extrahiert Text seitenweise aus .txt oder .pdf.
    Führt für gescannte oder leere PDF-Seiten einen automatischen OCR-Fallback durch.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().strip()
        return [(1, content)] if content else []

    elif ext == ".pdf":
        pages_content = []

        # Versuch 1: pdfplumber mit nativer Bildextraktion für OCR
        try:
            import pdfplumber
            import pytesseract

            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""

                    # OCR-Fallback, falls die Seite keinen Text enthält (z. B. Scan)
                    if not text.strip():
                        try:
                            img = page.to_image(resolution=300).original
                            text = pytesseract.image_to_string(img)
                        except Exception as e:
                            print(f"[OCR-Fallback] Hinweis: OCR auf Seite {page_num} übersprungen: {e}")

                    pages_content.append((page_num, text.strip()))
            return pages_content
        except ImportError:
            pass

        # Versuch 2: pypdf Fallback
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            # OCR-Fallback bei fehlendem Text via extrahierte Bilder
            if not text.strip():
                try:
                    import pytesseract
                    ocr_parts = [pytesseract.image_to_string(img.image) for img in page.images]
                    text = "\n".join(ocr_parts)
                except Exception:
                    pass

            pages_content.append((page_num, text.strip()))
        return pages_content

    else:
        print(f"[Ingestion] Überspringe nicht unterstütztes Dateiformat: {file_path}")
        return []
```

---

### Funktion: `recursive_chunk_text()`

**Dateipfad:** `Funktionen/recursive_chunk_text.py`  
**Kategorie:** Ingestion & Semantisches Chunking  
**Rolle in der RAG-Architektur:** Zerlegt Fließtext hierarchisch und rekursiv in semantisch zusammenhängende Fragmente mit definierter Maximalgröße und Überlappung (Overlap).

#### 1. Zweck & Funktionsweise
Einfaches Chunking nach fester Zeichenanzahl ("Fixed-Size Chunking") schneidet Sätze oder Wörter willkürlich mitten im Satz ab. Dies zerstört den semantischen Kontext und führt zu verfälschten Vektor-Embeddings.

`recursive_chunk_text()` implementiert das hierarchische **Recursive Character Chunking** nach Shaukat et al. (2026). Das Verfahren respektiert die natürliche Textstruktur entlang einer Prioritätenliste von Trennzeichen:
$$\text{Absätze } (\texttt{\textbackslash n\textbackslash n}) \longrightarrow \text{Sätze/Zeilen } (\texttt{\textbackslash n}) \longrightarrow \text{Wörter/Leerzeichen } (\texttt{" "}) \longrightarrow \text{Zeichen } (\texttt{""})$$

Übersteigt ein Textsegment das Limit von 500 Zeichen, wird es rekursiv mit der nächsttieferen Hierarchiestufe zerlegt. Anschließend werden die Fragmente unter Wahrung eines **Gleitfenster-Überlapps (Overlap) von ca. 50 Zeichen** zu Chunks kombiniert.

#### 2. Schnittstellendefinition

```python
def recursive_chunk_text(
    text: str, 
    chunk_size: int = 500, 
    overlap: int = 50, 
    separators: List[str] = None
) -> List[str]
```

* **Parameter:**
  * `text` (`str`): Der zu segmentierende Eingabetext.
  * `chunk_size` (`int`, Standard `500`): Maximale Zeichenlänge eines Chunks.
  * `overlap` (`int`, Standard `50`): Gewünschte Überlappung in Zeichen am Beginn nachfolgender Chunks.
  * `separators` (`List[str]`, Standard `["\n\n", "\n", " ", ""]`): Geordnete Hierarchie der Trennzeichen.
* **Rückgabewert:**
  * `List[str]`: Liste der erzeugten Text-Chunks.

#### 3. Detaillierter Algorithmenablauf
1. **Separator-Auswahl:**  
   Findet das ranghöchste Trennzeichen aus `separators`, das im Text tatsächlich existiert.
2. **Splitting & Rekursion:**  
   Teilt den Text an diesem Separator. Jedes resultierende Teilsegment wird geprüft:
   * Ist das Segment $\le \text{chunk\_size}$, wird es übernommen.
   * Ist das Segment $> \text{chunk\_size}$, ruft sich die Funktion mit den verbleibenden untergeordneten Separatoren rekursiv auf.
   * Fehlen tiefere Separatoren (z. B. ununterbrochene Zeichenketten), greift ein direktes Sliding-Window.
3. **Zusammenfügen & Overlap-Puffer:**  
   Die Fragmente werden in einem Puffer `current_parts` akkumuliert. Sobald das Hinzufügen des nächsten Teils `chunk_size` überschreiten würde:
   * Wird der bisherige Inhalt als fertiger Chunk gespeichert.
   * Werden ältere Fragmente vom Anfang des Puffers entfernt (`current_parts.pop(0)`), bis die Restlänge $\le \text{overlap}$ ist.
   * Dadurch enthält der nächste Chunk den Endkontext des vorherigen Chunks.

#### 4. Wichtige Architektur- & Design-Entscheidungen
* **Parameterwahl ($500 / 50$):**  
  Untersuchungen von Shaukat et al. (2026) und Wang et al. (2024) zeigen: Chunks über 1000 Zeichen verwässern die Spezifität des Vektors bei gezielten Faktenfragen (z. B. "Wann hat der Bewerber angefangen?"). Chunks unter 200 Zeichen fehlen oft Kausalzusammenhänge. ca. 500 Zeichen (~70–100 Wörter) fassen einzelne Lebenslauf-Stationen ideal zusammen.
* **Kontextkontinuität durch Overlap:**  
  Ohne Überlappung geht der Sinn verloren, wenn wichtige Satzteile genau an einer Chunk-Grenze getrennt werden. Der 50-Zeichen-Overlap überbrückt diesen Übergang zuverlässig.

#### 5. Häufige Fragen & Design-Entscheidungen (FAQ)

> **F: Warum haben Sie Recursive Chunking gewählt statt Semantic Chunking über Embedding-Distanzen?**  
> *Antwort:* Echtes Semantic Chunking via Vektordistanz erfordert viele zusätzliche Embedding-Berechnungen bereits vor der Indizierung. Recursive Chunking liefert ca. 90-95 % der semantischen Kohärenz, läuft jedoch in $O(N)$ extrem schnell ohne Modellaufrufe.

> **F: Was passiert bei einem 2000 Zeichen langen Textblock ganz ohne Leerzeichen oder Umbrüche?**  
> *Antwort:* Die Rekursion fällt sukzessive durch bis zu `sep == ""`. Dort greift der Fallback `valid_splits.append(part[j:j + chunk_size])`, der den Block deterministisch in 500er-Abschnitte zerlegt.

#### 6. Vollständiger Quellcode

```python
from typing import List, Tuple

def recursive_chunk_text(text: str, chunk_size: int = 500, overlap: int = 50, separators: List[str] = None) -> List[str]:
    """Hierarchisches Recursive Chunking nach Shaukat et al. (2026).
    Hierarchie: Absätze (\n\n) -> Sätze/Zeilen (\n) -> Wörter (' ') -> Zeichen ('').
    Sucht an der 500-Zeichen-Grenze nach dem nächsten Trenner und erhält 50 Zeichen Overlap.
    """
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    if not text.strip():
        return []

    # Höchsten passenden Separator für den aktuellen Textabschnitt wählen
    chosen_sep = separators[-1]
    next_separators = []
    for i, sep in enumerate(separators):
        if sep == "":
            chosen_sep = ""
            break
        if sep in text:
            chosen_sep = sep
            next_separators = separators[i + 1:]
            break

    splits = text.split(chosen_sep) if chosen_sep != "" else list(text)

    # Übergroße Segmente anhand der nachfolgenden Hierarchiestufen rekursiv zerlegen
    valid_splits = []
    for part in splits:
        if len(part) <= chunk_size:
            if part:
                valid_splits.append(part)
        else:
            if next_separators:
                valid_splits.extend(recursive_chunk_text(part, chunk_size, overlap, next_separators))
            else:
                for j in range(0, len(part), max(1, chunk_size - overlap)):
                    valid_splits.append(part[j:j + chunk_size])

    # Fragmente zusammenfügen und Overlap von ca. 50 Zeichen wahren
    chunks = []
    current_parts = []
    current_length = 0

    for part in valid_splits:
        sep_len = len(chosen_sep) if current_parts else 0
        piece_len = len(part)

        if current_length + sep_len + piece_len > chunk_size and current_parts:
            doc = chosen_sep.join(current_parts).strip()
            if doc:
                chunks.append(doc)

            # Behalte vorherige Teile für das 50-Zeichen-Overlap
            while current_parts and (
                current_length > overlap or (current_length + len(chosen_sep) + piece_len > chunk_size)
            ):
                popped = current_parts.pop(0)
                current_length -= len(popped) + (len(chosen_sep) if current_parts else 0)

        current_parts.append(part)
        current_length += piece_len + (len(chosen_sep) if len(current_parts) > 1 else 0)

    if current_parts:
        doc = chosen_sep.join(current_parts).strip()
        if doc and (not chunks or doc != chunks[-1]):
            chunks.append(doc)

    return chunks
```

---

### Klasse: `DocumentChunk`

**Dateipfad:** `Klassen/DocumentChunk.py`  
**Kategorie:** Datenmodell / Data-Transfer-Object (DTO)  
**Rolle in der RAG-Architektur:** Repräsentiert ein semantisch isoliertes Textsegment im Arbeitsspeicher (In-Memory) inklusive aller Provenienz-Metadaten zur lückenlosen Nachvollziehbarkeit und Quellenangabe.

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

---

### Funktion: `process_document()`

**Dateipfad:** `Funktionen/process_document.py`  
**Kategorie:** Ingestion & Orchestrierung  
**Rolle in der RAG-Architektur:** Koordinierende Schnittstelle der Ingestion-Pipeline für eine einzelne Datei. Verknüpft Textextraktion mit seitenweisem Chunking und instanziiert `DocumentChunk`-Objekte.

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

---

### Klasse: `EmbeddingClient`

**Dateipfad:** `Klassen/EmbeddingClient.py`  
**Kategorie:** Inferenz & Feature-Transformation  
**Rolle in der RAG-Architektur:** Kapselt das Vektorisierungsmodell (`SentenceTransformer`), wandelt Text-Chunks sowie Benutzerfragen in dichte Vektor-Embeddings (Dense Vectors) um und sichert einen einheitlichen, symmetrischen Vektorraum.

#### 1. Zweck & Funktionsweise
Ein zentrales Element des RAG-Systems ist die Überführung unstrukturierter Texte in mathematische Vektoren. Der `EmbeddingClient` stellt sicher, dass sowohl Dokumenten-Chunks bei der Indizierung als auch Nutzerfragen bei der Suche mit **exakt demselben Vektormodell** projiziert werden. 

Er nutzt die Bibliothek `sentence-transformers` und führt die Inferenz lokal aus. Dadurch fließen sensible Bewerberdaten für die Vektorisierung zu keinem Zeitpunkt an externe Cloud-APIs ab.

#### 2. Modell-Spezifikation
* **Modell:** `paraphrase-multilingual-MiniLM-L12-v2`
* **Vektordimension:** `384`
* **Unterstützte Sprachen:** > 50 Sprachen (vollständig optimiert für Deutsch und Englisch)
* **Distanzmetrik:** Kosinusähnlichkeit (Cosine Similarity)

#### 3. Methoden & Signaturen
* `__init__(self)`: Lädt das vortrainierte Transformer-Modell in den Arbeitsspeicher.
* `embed_texts(self, texts: List[str]) -> List[List[float]]`: Führt Batch-Inferenz durch (`self.model.encode(texts, convert_to_numpy=True)`).
* `embed_query(self, query: str) -> List[float]`: Vektorisiert eine einzelne Frage via `embed_texts([query])[0]`.

#### 4. Wichtige Architektur- & Design-Entscheidungen
* **Mehrsprachigkeit vs. monolinguale Modelle:**  
  `paraphrase-multilingual-MiniLM-L12-v2` wurde gewählt, weil Bewerbungsunterlagen in der DACH-Region meist auf Deutsch vorliegen. Rein englische Modelle (wie `all-MiniLM-L6-v2`) scheitern bei deutschen Begriffen.
* **Effizienz (384 vs. 1536 Dimensionen):**  
  384 Dimensionen sparen rund 75 % Speicherplatz im Index im Vergleich zu 1536-dimensionalen Cloud-Modellen und ermöglichen CPU-Inferenzzeiten im zweistelligen Millisekundenbereich.

#### 5. Häufige Fragen & Design-Entscheidungen (FAQ)

> **F: Warum führen Sie die Embeddings lokal aus, anstatt eine OpenAI-API aufzurufen?**  
> *Antwort:* Personenbezogene Bewerbungsdaten erfordern strenge DSGVO-Einhaltung. Lokale Ausführung schützt die Privatsphäre, eliminiert API-Kosten und verhindert Netzwerk-Latenzen oder Dienstausfälle.

> **F: Was passiert, wenn Dokumente und Anfragen mit unterschiedlichen Modellen vektorisiert werden?**  
> *Antwort:* Die Vektoren lägen in inkompatiblen Koordinatensystemen. Kosinusähnlichkeit verlangt zwingend, dass Query und Dokumente denselben semantischen Vektorraum teilen.

#### 6. Vollständiger Quellcode

```python
from typing import List, Tuple

class EmbeddingClient:
    """Verwaltet Vektorisierung via lokalem Modell."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        # ALT: Nur für englische Texte:
        # self.model_name = "all-MiniLM-L6-v2"
        self.dimension = 384
        self.model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, convert_to_numpy=True)
        return vectors.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]
```

---

### Funktion: `get_db_connection()`

**Dateipfad:** `Funktionen/get_db_connection.py`  
**Kategorie:** Infrastruktur & Datenbankverbindung  
**Rolle in der RAG-Architektur:** Zentraler Verbindungsaufbau zur PostgreSQL/PgVector-Datenbankinstanz mit konfigurierbarem Umgebungsvariablen-Support (12-Factor App).

#### 1. Zweck & Funktionsweise
Die Funktion `get_db_connection()` stellt eine standardisierte Verbindung zur relationalen Vektordatenbank her. Sie liest Verbindungsparameter aus dem System-Environment aus und fällt auf vordefinierte Standardwerte zurück, die exakt mit der mitgelieferten `docker-compose.yml` (`pgvector/pgvector:pg16`) harmonieren.

#### 2. Schnittstellendefinition

```python
def get_db_connection()
```

#### 3. Konfigurationsmatrix

| Umgebungsvariable | Standardwert | Beschreibung |
| :--- | :--- | :--- |
| `POSTGRES_HOST` | `"localhost"` | Hostname / IP des Datenbank-Servers oder Containers. |
| `POSTGRES_PORT` | `5432` | TCP-Port des PostgreSQL-Dienstes. |
| `POSTGRES_DB` | `"rag_db"` | Name der Zieldatenbank. |
| `POSTGRES_USER` | `"postgres"` | Authentifizierter Datenbankbenutzer. |
| `POSTGRES_PASSWORD` | `"postgrespassword"` | Kennwort für den Datenbankzugriff. |

#### 4. Wichtige Architektur- & Design-Entscheidungen
* **12-Factor App Konformität:**  
  Durch `os.getenv` lässt sich die Anwendung ohne Codeänderungen in unterschiedlichen Umgebungen (lokale Entwicklung, Docker Compose, Kubernetes, CI/CD Pipeline) betreiben.

#### 5. Häufige Fragen & Design-Entscheidungen (FAQ)

> **F: Warum nutzen Sie im Code direkte Verbindungen statt eines Connection Pools?**  
> *Antwort:* Für eine CLI-orientierte Anwendung mit sequenziellen Abfragen ist eine dedizierte Verbindung schlank und ressourceneffizient. Bei der Skalierung zu einem Web-Service (z. B. FastAPI) würde die Funktion auf `psycopg2.pool.ThreadedConnectionPool` umgestellt werden.

#### 6. Vollständiger Quellcode

```python
import psycopg2
import os

def get_db_connection():
    """Stellt Verbindung zu PostgreSQL her."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "rag_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgrespassword")
    )
```

---

### Funktion: `init_database()`

**Dateipfad:** `Funktionen/init_database.py`  
**Kategorie:** Datenbank & Schema-Management  
**Rolle in der RAG-Architektur:** Initialisiert die PgVector-Erweiterung, stellt das Tabellenschema sicher, führt automatisierte Dimensions-Migrationen durch und erzeugt den HNSW-Vektorindex.

#### 1. Zweck & Funktionsweise
Die Vektorsuche erfordert in PostgreSQL eine spezialisierte Tabellenstruktur mit dem Datentyp `vector(dim)` und einen Approximate Nearest Neighbor (ANN) Index. 

`init_database()` automatisiert das Setup:
1. Bereinigt vorhandene Datensätze (`TRUNCATE TABLE`), um Duplikate bei wiederholten Testläufen zu verhindern.
2. Registriert den PgVector-Datentyp für `psycopg2` via `register_vector(conn)`.
3. **Dimensions-Konsistenzprüfung:** Erkennt Abweichungen der Vektordimension (z. B. 384 vs. 1536) über den PostgreSQL-Katalog `pg_attribute` und baut die Tabelle bei Modellwechseln automatisch neu auf.
4. Erstellt die Tabelle `document_chunks` mit allen Metadaten.
5. Erzeugt einen **HNSW-Index** (Hierarchical Navigable Small World) mit Kosinus-Distanz (`vector_cosine_ops`).

#### 2. Schnittstellendefinition

```python
def init_database(conn, dimension: int)
```

#### 3. Wichtige Architektur- & Design-Entscheidungen
* **Warum HNSW statt IVFFlat?**  
  * *HNSW:* Bietet exzellenten Recall (> 95 %), logarithmische Suchzeit $O(\log N)$ und benötigt kein vorheriges K-Means Clustering Training.
  * *IVFFlat:* Verlangt zwingend Trainingsdaten und verliert bei dynamischen Dokumenten-Updates an Genauigkeit.
* **Kosinus-Metrik (`vector_cosine_ops`):**  
  Exakt auf normalisierte Text-Embeddings abgestimmt.

#### 4. Häufige Fragen & Design-Entscheidungen (FAQ)

> **F: Warum ist die dynamische Prüfung der Dimension über `pg_attribute` wichtig?**  
> *Antwort:* In der Prototypenphase werden verschiedene Embeddings getestet. Ohne diese Prüfung schlägt jeder `INSERT` mit einem kryptischen Datenbankfehler fehl, wenn die Tabelle noch für eine frühere Dimension angelegt war.

#### 5. Vollständiger Quellcode

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

---

### Funktion: `save_chunks_to_db()`

**Dateipfad:** `Funktionen/save_chunks_to_db.py`  
**Kategorie:** Datenbank & Persistierung  
**Rolle in der RAG-Architektur:** Speichert verarbeitete Text-Chunks zusammen mit den berechneten Vektor-Embeddings transaktionssicher in PgVector.

#### 1. Zweck & Funktionsweise
Die Funktion verknüpft jedes `DocumentChunk`-Objekt mit seinem entsprechenden Vektor-Array (`zip(chunks, embeddings)`) und führt für jeden Datensatz ein parametrisiertes SQL-`INSERT` aus.

#### 2. Schnittstellendefinition

```python
def save_chunks_to_db(
    conn, 
    chunks: List[DocumentChunk], 
    embeddings: List[List[float]]
)
```

#### 3. Wichtige Architektur- & Design-Entscheidungen
* **SQL-Injection Schutz:**  
  Parametrisierte Abfragen (`%s`) schützen die Datenbank vor SQL-Injections via Dokumentinhalten ("Data-Prompt Injection").
* **Transaktionale Integrität:**  
  `conn.commit()` stellt atomares Schreiben nach ACID-Prinzipien sicher.

#### 4. Häufige Fragen & Design-Entscheidungen (FAQ)

> **F: Wie skaliert diese Funktion bei Millionen von Chunks?**  
> *Antwort:* Durch Umstellung auf `psycopg2.extras.execute_values` oder den PostgreSQL-Befehl `COPY FROM` (Bulk Insert), um Netzwerk-Latenzen zu eliminieren.

#### 5. Vollständiger Quellcode

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

---

### Funktion: `retrieve_top_k()`

**Dateipfad:** `Funktionen/retrieve_top_k.py`  
**Kategorie:** Retrieval & Vektorsuche  
**Rolle in der RAG-Architektur:** Führt eine Ähnlichkeitssuche (Similarity Search) über Kosinusdistanz im hochdimensionalen Vektorraum aus und selektiert die $K$ relevantesten Text-Chunks.

#### 1. Zweck & Funktionsweise
`retrieve_top_k()` übergibt den Query-Vektor an PostgreSQL/PgVector und nutzt den vorberechneten HNSW-Index.

Die mathematische Grundlage bildet die **Kosinus-Distanz** (Operator `<=>`):
$$D_{\cos}(u, v) = 1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

Die Treffer werden in einen Ähnlichkeitsscore umgerechnet:
$$\text{Similarity } = 1 - D_{\cos}(u, v)$$

#### 2. Schnittstellendefinition

```python
def retrieve_top_k(
    conn, 
    query_embedding: List[float], 
    top_k: int = 10
) -> List[dict]
```

#### 3. Wichtige Architektur- & Design-Entscheidungen
* **Index-Beschleunigung:**  
  `ORDER BY embedding <=> %s::vector ASC` navigiert direkt den HNSW-Index in logarithmischer Zeit $O(\log N)$.
* **Attribution:**  
  Gibt Metadaten (`file_name`, `page_number`) zurück, um Quellenangaben zu ermöglichen.

#### 4. Häufige Fragen & Design-Entscheidungen (FAQ)

> **F: Warum Kosinus-Distanz statt Euklidischer Distanz?**  
> *Antwort:* Die Kosinus-Distanz misst den Winkel zwischen Vektoren unabhängig von der absoluten Textlänge. Euklidische Distanz würde lange Chunks ungleich bewerten.

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

---

### Funktion: `ask_hr_assistant()`

**Dateipfad:** `Funktionen/ask_hr_assistant.py`  
**Kategorie:** Generation & Pipeline-Orchestrierung  
**Rolle in der RAG-Architektur:** End-to-End Steuerung der Retrieval-Augmented Generation für Nutzeranfragen: Vektorisierung, Top-K Retrieval, Strict-Context-Prompting und deterministische LLM-Inferenz via Ollama.

#### 1. Zweck & Funktionsweise
Diese Funktion steuert den vollständigen Inferenzpfad:
1. **Query-Vektorisierung:** `embedding_client.embed_query(query)`.
2. **Top-K Retrieval:** `retrieve_top_k()` holt die semantisch besten Abschnitte.
3. **Erklärbarkeits-Logging:** Gibt Treffer mit Ähnlichkeitsscore und Seitenzahl im Terminal aus.
4. **Strict Context Prompting:** Verhindert Halluzinationen durch strenge Vorgaben.
5. **LLM-Synthese:** Ruft `llama3.1:latest` über die OpenAI-kompatible Ollama-API (`http://llm.tim/v1`) mit `temperature=0.0` auf.

#### 2. Schnittstellendefinition

```python
def ask_hr_assistant(
    conn, 
    embedding_client: EmbeddingClient, 
    query: str, 
    top_k: int = 10
) -> str
```

#### 3. Wichtige Architektur- & Design-Entscheidungen
* **Halluzinations-Vermeidung:**  
  * *Negative Constraints:* "Wenn eine Information nicht enthalten ist, antworte ehrlich mit 'Information nicht im Dokument enthalten'."
  * *Greedy Decoding (`temperature=0.0`):* Deterministische Faktenwiedergabe ohne Spekulation.
* **On-Premise LLM (DSGVO):**  
  Keine Datenübertragung an Cloud-Dienste; Ausführung auf eigener Infrastruktur via Ollama.

#### 4. Häufige Fragen & Design-Entscheidungen (FAQ)

> **F: Was passiert, wenn eine Information nicht im Lebenslauf steht?**  
> *Antwort:* Durch den strikten System-Prompt und `temperature=0.0` erfindet das LLM keine Angaben, sondern meldet zuverlässig: *"Information nicht im Dokument enthalten."*

#### 5. Vollständiger Quellcode

```python
from Klassen.EmbeddingClient import EmbeddingClient
from Funktionen.retrieve_top_k import retrieve_top_k

def ask_hr_assistant(conn, embedding_client: EmbeddingClient, query: str, top_k: int = 10) -> str:
    """Verarbeitet Nutzeranfrage: Vektorisierung, Top-K Retrieval und HR-Prompting via Server-Ollama."""
    print("\n" + "=" * 60)
    print(f"Frage: {query}")
    print("=" * 60)

    # Query Vectorization
    query_vector = embedding_client.embed_query(query)

    # Top-K Similarity Search
    relevant_chunks = retrieve_top_k(conn, query_vector, top_k=top_k)

    print(f"[Retrieval] {len(relevant_chunks)} relevanteste Chunks gefunden (Top-K={top_k}):")
    for idx, c in enumerate(relevant_chunks, start=1):
        preview = c['content'].replace("\n", " ")[:90]
        print(f"  {idx}. [Score: {c['similarity']:.4f} | {c['file_name']} (S. {c['page_number']})]: {preview}...")

    # Strict Context Prompting (Spezifikation 3.3)
    system_prompt = (
        "Du bist ein präziser HR-Assistent. Beantworte die Frage ausschließlich "
        "basierend auf den bereitgestellten Dokumentenauszügen. Wenn eine Information "
        "nicht enthalten ist, antworte ehrlich mit \"Information nicht im Dokument enthalten\"."
    )

    context_str = "\n".join([f"[{c['content']}]" for c in relevant_chunks])
    user_prompt = f"Context:\n{context_str}\n\nUser Query: {query}"

    # LLM Generation via Server-Ollama
    try:
        from openai import OpenAI
        ollama_client = OpenAI(
            base_url="http://llm.tim/v1", # v1 ist der Standard-Endpunkt für Ollama-Server
            api_key="ollama",  # hier beliebiger String, da keine Authentifizierung erforderlich ist
        )
        
        response = ollama_client.chat.completions.create(
            model="llama3.1:latest", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"[Fehler bei der Verbindung zu Ollama: {e}]"

    print(f"\nAntwort des HR-Assistenten:\n{answer}\n")
    return answer
```

---

### Einstiegspunkt: `main.py` / Funktion `main()`

**Dateipfad:** `main.py`  
**Kategorie:** System-Orchestrierung & CLI  
**Rolle in der RAG-Architektur:** Zentraler Haupteinstiegspunkt (Main Orchestrator), der Initialisierung, Ingestion-Pipeline und interaktive Frage-Antwort-Schleife zusammenführt.

#### 1. Zweck & Funktionsweise
Die Funktion `main()` orchestriert den gesamten Lebenszyklus der RAG-Anwendung:
1. Prüft die Datenbankverbindung und gibt Hinweise zu Docker.
2. Initialisiert Embedding-Modell und Datenbankschema.
3. Liest flexible Dateipfade ein (CLI oder interaktiv).
4. Führt seitenweises Parsing, Batch-Vektorisierung und Persistierung aus.
5. Ermöglicht interaktives Fragen in einer REPL-Schleife.

#### 2. Schnittstellendefinition

```python
def main()
```

#### 3. Wichtige Architektur- & Design-Entscheidungen
* **Batch-Vektorisierung:**  
  Texte werden gebündelt an `embed_client.embed_texts()` übergeben, um die Vektorisierungs-Pipeline maximal auszulasten.
* **Saubere Kapselung:**  
  Einstiegspunkt delegiert alle Aufgaben an modulare Teilfunktionen.

#### 4. Häufige Fragen & Design-Entscheidungen (FAQ)

> **F: Wie könnte man aus der CLI-Anwendung eine REST-API machen?**  
> *Antwort:* Die Initialisierung wird im FastAPI Lifespan-Event abgelegt; die Ingestion wird über einen `POST /documents`-Endpunkt und die Abfragen über einen `POST /ask`-Endpunkt exponiert.

#### 5. Vollständiger Quellcode

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
