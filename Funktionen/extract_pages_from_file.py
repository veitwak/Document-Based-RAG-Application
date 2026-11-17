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