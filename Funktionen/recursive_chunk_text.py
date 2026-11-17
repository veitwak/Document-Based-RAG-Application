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
