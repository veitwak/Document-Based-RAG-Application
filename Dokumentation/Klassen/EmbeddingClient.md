### Klasse: `EmbeddingClient`

**Dateipfad:** `Klassen/EmbeddingClient.py`  
**Kategorie:** Inferenz & Feature-Transformation  
**Rolle in der RAG-Architektur:** Kapselt das Vektorisierungsmodell (`SentenceTransformer`), wandelt Text-Chunks sowie Benutzerfragen in dichte Vektor-Embeddings (Dense Vectors) um und sichert einen einheitlichen, symmetrischen Vektorraum.

---

#### 1. Zweck & Funktionsweise
Ein zentrales Element des RAG-Systems ist die Überführung unstrukturierter Texte in mathematische Vektoren. Der `EmbeddingClient` stellt sicher, dass sowohl Dokumenten-Chunks bei der Indizierung als auch Nutzerfragen bei der Suche mit **exakt demselben Vektormodell** projiziert werden. 

Er nutzt die Bibliothek `sentence-transformers` und führt die Inferenz lokal aus. Dadurch fließen sensible Bewerberdaten für die Vektorisierung zu keinem Zeitpunkt an externe Cloud-APIs ab.

#### 2. Modell-Spezifikation

* **Modell:** `paraphrase-multilingual-MiniLM-L12-v2`
* **Vektordimension:** `384`
* **Unterstützte Sprachen:** > 50 Sprachen (vollständig optimiert für Deutsch und Englisch)
* **Distanzmetrik:** Kosinusähnlichkeit (Cosine Similarity)

#### 3. Methoden & Signaturen

##### `__init__(self)`
* Lädt das vortrainierte Transformer-Modell in den Arbeitsspeicher.
* Setzt die Vektordimension (`self.dimension = 384`).
* *Hinweis:* Nutzt Lazy-Import für `sentence_transformers`, damit das Modul nur geladen wird, wenn tatsächlich instanziiert wird.

##### `embed_texts(self, texts: List[str]) -> List[List[float]]`
* **Parameter:** `texts`: Liste von Strings (Textfragmente).
* **Rückgabe:** Liste von 384-dimensionalen Fließkomma-Vektoren (`List[List[float]]`).
* **Besonderheit:** Führt Batch-Inferenz durch (`self.model.encode(texts, convert_to_numpy=True)`). Nutzt CPU/GPU-Parallelisierung für maximalen Durchsatz und konvertiert das NumPy-Array via `.tolist()` in native Python-Listen für `psycopg2`.

##### `embed_query(self, query: str) -> List[float]`
* **Parameter:** `query`: Einzelner Suchbegriff / Frage des Nutzers.
* **Rückgabe:** Ein 384-dimensionaler Vektor (`List[float]`).
* **Besonderheit:** Ruft intern `embed_texts([query])[0]` auf, um Codelogik redundant zu vermeiden.

#### 4. Wichtige Architektur- & Design-Entscheidungen
* **Mehrsprachigkeit vs. monolinguale Modelle:**  
  Frühere Entwürfe nutzten `all-MiniLM-L6-v2`. Dieses ist jedoch rein auf die englische Sprache trainiert. Da Bewerbungsunterlagen in der DACH-Region fast ausschließlich auf Deutsch vorliegen, führte das zu semantischen Fehlzuordnungen. `paraphrase-multilingual-MiniLM-L12-v2` bildet deutsche Formulierungen und Fragen präzise im semantischen Vektorraum ab.
* **Effizienz (384 vs. 1536 Dimensionen):**  
  Modelle wie OpenAIs `text-embedding-3-small` liefern 1536 Dimensionen. Die 384 Dimensionen von MiniLM sparen ca. 75 % RAM- und Festplattenspeicher im HNSW-Index von PgVector und ermöglichen schnelle Inferenzzeiten (< 100 ms) auf Standard-CPUs.

#### 5. Häufige Fragen & Design-Entscheidungen (FAQ)

> **F: Warum führen Sie die Embeddings lokal aus, anstatt eine OpenAI-API aufzurufen?**  
> *Antwort:* Bewerbungsdaten (Name, Adresse, Zeugnisnoten) sind personenbezogene Daten nach DSGVO. Lokale Embeddings garantieren vollständige Datenhoheit, Unabhängigkeit von API-Ratenbegrenzungen und eliminiertes Risiko von Netzwerk-Timeouts.

> **F: Was passiert, wenn Dokumente und Anfragen mit unterschiedlichen Modellen vektorisiert werden?**  
> *Antwort:* Symmetrische Vektorsuche funktioniert nur, wenn Dokumente und Queries denselben semantischen Vektorraum teilen. Unterschiedliche Modelle besitzen völlig andere Koordinatensysteme – eine Kosinusähnlichkeit wäre mathematisch sinnlos.

---

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
