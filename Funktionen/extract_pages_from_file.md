### Funktion: `extract_pages_from_file()`

**Dateipfad:** `Funktionen/extract_pages_from_file.py`  
**Kategorie:** Ingestion & Dokumenten-Extraktion  
**Rolle in der RAG-Architektur:** Universeller Extraktor für `.txt`- und `.pdf`-Dateien mit automatischem Multi-Tier OCR-Fallback für gescannte oder rein bildbasierte Dokumentenseiten.

---

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

---

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
