### Funktion: `get_db_connection()`

**Dateipfad:** `Funktionen/get_db_connection.py`  
**Kategorie:** Infrastruktur & Datenbankverbindung  
**Rolle in der RAG-Architektur:** Zentraler Verbindungsaufbau zur PostgreSQL/PgVector-Datenbankinstanz mit konfigurierbarem Umgebungsvariablen-Support (12-Factor App).

---

#### 1. Zweck & Funktionsweise
Die Funktion `get_db_connection()` stellt eine standardisierte Verbindung zur relationalen Vektordatenbank (PostgreSQL mit `pgvector`) her. 

Sie liest Verbindungsparameter aus dem System-Environment aus und fällt auf vordefinierte Standardwerte zurück, die exakt mit der mitgelieferten `docker-compose.yml` (`pgvector/pgvector:pg16`) harmonieren.

#### 2. Schnittstellendefinition

```python
def get_db_connection()
```

* **Parameter:** Keine.
* **Rückgabewert:** Eine aktive `psycopg2.extensions.connection`-Instanz.

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
  Hardcodierte Verbindungsdaten sind ein Sicherheits- und Wartungsrisiko. Durch `os.getenv` lässt sich die Anwendung ohne Codeänderungen in unterschiedlichen Umgebungen (lokale Entwicklung, Docker Compose, Kubernetes, CI/CD Pipeline) betreiben.
* **Entkopplung:**  
  Alle anderen Module (`main.py`, `init_database.py`, `save_chunks_to_db.py`, `retrieve_top_k.py`) erhalten die geöffnete Verbindung per Dependency Injection, anstatt eigene Verbindungskonfigurationen zu verwalten.

#### 5. Häufige Fragen & Design-Entscheidungen (FAQ)

> **F: Warum nutzen Sie im Code direkte Verbindungen statt eines Connection Pools?**  
> *Antwort:* Für eine CLI-orientierte Anwendung mit sequenziellen Abfragen ist eine dedizierte Verbindung schlank und ressourceneffizient. Bei der Skalierung zu einem Web-Service (z. B. FastAPI) würde die Funktion auf `psycopg2.pool.ThreadedConnectionPool` oder einen asynchronen Treiber wie `asyncpg` umgestellt werden.

---

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
