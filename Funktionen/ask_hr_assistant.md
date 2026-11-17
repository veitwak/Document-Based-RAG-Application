### Funktion: `ask_hr_assistant()`

**Dateipfad:** `Funktionen/ask_hr_assistant.py`  
**Kategorie:** Generation & Pipeline-Orchestrierung  
**Rolle in der RAG-Architektur:** End-to-End Steuerung der Retrieval-Augmented Generation für Nutzeranfragen: Vektorisierung, Top-K Retrieval, Strict-Context-Prompting und deterministische LLM-Inferenz via Ollama.

---

#### 1. Zweck & Funktionsweise
Diese Funktion repräsentiert die Ausführungsphase des RAG-Systems. Sie nimmt eine in natürlicher Sprache formulierte Benutzerfrage entgegen und führt alle erforderlichen Schritte bis zur fertigen Antwort aus:
1. **Query-Vektorisierung:** Die Frage wird über den `EmbeddingClient` in denselben semantischen Raum transformiert wie die Dokumente.
2. **Top-K Retrieval:** `retrieve_top_k()` holt die semantisch relevantesten Textauszüge aus der PgVector-Datenbank.
3. **Erklärbarkeits-Logging:** Die gefundenen Treffer werden inklusive Ähnlichkeitsscore, Dateiname und Seitenzahl auf der Konsole ausgegeben.
4. **Strict Context Prompting:** Der Kontext wird mit Sicherheitsanweisungen ("Guardrails") in einen Prompt eingebettet.
5. **LLM-Synthese:** Das Sprachmodell `llama3.1:latest` wird über eine lokale Ollama-Schnittstelle (`http://llm.tim/v1`) mit `temperature=0.0` aufgerufen, um Halluzinationen zu unterbinden.

#### 2. Schnittstellendefinition

```python
def ask_hr_assistant(
    conn, 
    embedding_client: EmbeddingClient, 
    query: str, 
    top_k: int = 10
) -> str
```

* **Parameter:**
  * `conn`: Aktive PostgreSQL-Verbindung.
  * `embedding_client` (`EmbeddingClient`): Die Instanz des Embedding-Modells.
  * `query` (`str`): Die Frage des Benutzers (z. B. "Welche Fähigkeiten werden genannt?").
  * `top_k` (`int`, Standard `10`): Anzahl der Textfragmente, die dem LLM als Kontext bereitgestellt werden.
* **Rückgabewert:**
  * `str`: Die vom LLM generierte finale Antwort.

#### 3. Das Prompt-Design ("Strict Context Prompting")

```text
System: Du bist ein präziser HR-Assistent. Beantworte die Frage ausschließlich 
basierend auf den bereitgestellten Dokumentenauszügen. Wenn eine Information 
nicht enthalten ist, antworte ehrlich mit "Information nicht im Dokument enthalten".

Context:
[Chunk 1]
[Chunk 2]
...
[Chunk K]

User Query: {query}
```

#### 4. Wichtige Architektur- & Design-Entscheidungen
* **Verhinderung von Halluzinationen:**  
  * *Negative Constraints:* Der System-Prompt enthält eine explizite Abbruchbedingung ("Information nicht im Dokument enthalten"). Fehlt die Information im Kontext, darf das Modell nicht fabrizieren.
  * *Greedy Decoding (`temperature=0.0`):* Unterdrückt stochastische Zufallsvariationen bei der Tokenauswahl. Antworten bleiben konsistent, deterministisch und reproduzierbar.
* **DSGVO & Datenschutz via On-Premise LLM:**  
  Durch den Einsatz von Ollama mit Meta Llama 3.1 auf eigener Infrastruktur verlassen keine Bewerberdaten das interne Netzwerk.
* **OpenAI-kompatibles SDK:**  
  Die Nutzung der Standard-`openai`-Python-Bibliothek mit benutzerdefinierter `base_url="http://llm.tim/v1"` ermöglicht einen nahtlosen Modell- und Anbieterwechsel ohne Code-Refactoring.

#### 5. Häufige Fragen & Design-Entscheidungen (FAQ)

> **F: Was passiert, wenn eine Frage gestellt wird, zu der es im Dokument gar keine Antwort gibt (z. B. Führerschein)?**  
> *Antwort:* PgVector liefert zwar die rechnerisch ähnlichsten Chunks (allerdings mit niedrigem Ähnlichkeitsscore). Durch den strikten System-Prompt und `temperature=0.0` erkennt Llama 3.1 jedoch, dass der geforderte Fakt im Kontext fehlt, und antwortet wahrheitsgemäß mit: *"Information nicht im Dokument enthalten."*

> **F: Könnte man zusätzlich ein Reranking einbauen?**  
> *Antwort:* Ja. Nach dem Abruf der Top-20 Chunks aus PgVector könnte ein Cross-Encoder Reranker (z. B. `bge-reranker-large` oder `Cohere Rerank`) eingesetzt werden, um die semantische Reihenfolge weiter zu schärfen, bevor die Top-5 an das LLM übergeben werden.

---

#### 6. Vollständiger Quellcode

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
