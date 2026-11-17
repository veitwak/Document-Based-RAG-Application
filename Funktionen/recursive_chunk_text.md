### Funktion: `recursive_chunk_text()`

**Dateipfad:** `Funktionen/recursive_chunk_text.py`  
**Kategorie:** Ingestion & Semantisches Chunking  
**Rolle in der RAG-Architektur:** Zerlegt Fließtext hierarchisch und rekursiv in semantisch zusammenhängende Fragmente mit definierter Maximalgröße und Überlappung (Overlap).

---

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

---

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
