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
        import os
        from openai import OpenAI

        base_url = os.getenv("OLLAMA_BASE_URL", "http://llm.tim/v1")
        model_name = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

        ollama_client = OpenAI(
            base_url=base_url,  # v1 ist der Standard-Endpunkt für Ollama-Server
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),  # beliebiger String für lokale Instanzen
        )
        
        response = ollama_client.chat.completions.create(
            model=model_name, 
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
