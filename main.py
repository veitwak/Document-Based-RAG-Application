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

    '''
    # Beispielfragen zur Demonstration
    sample_queries = [
        "Wie lautet der Name des Bewerbers?",
        "Welche Fähigkeiten werden im Lebenslauf genannt?",
        "Wann hat der Bewerber seinen letzten Job begonnen?",
        "Welchen Führerschein besitzt der Bewerber?",
        "Wie lautet die Telefonnummer des Bewerbers und was ist seine Körpergröße?",
        "Ist der Bewerber deutscher Staatsbürger?"
    ]
    '''

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
