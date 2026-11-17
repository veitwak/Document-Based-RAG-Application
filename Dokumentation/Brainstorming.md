# Ziel

### Eine Anwendung entwickeln, die es einem Benutzer ermöglicht, Dokumente (z. B. einen Lebenslauf) hochzuladen und anschließend Fragen zu diesen Dokumenten zu stellen

## Inputformat

### Der Benutzer soll ein oder mehrere Dokumente hochladen können, z. B. einen Lebenslauf im PDF- oder Textformat sowie weitere relevante Dokumente

## Ablauf
### 1. Dokumente einlesen (nach upload)

#### 1.1 1 bis n Dokumente. Format ist entweder .pdf oder .txt
##### 1.1.1 .txt und .pdx mit textelementen --> Sollte problemlos gehen daraus den text zu extrahieren
##### 1 oder 2 py-Funktionen mit einem return. try except nutzen um den Dateiupload zu kontrollieren (.pdf oder .txt)

##### 1.1.2 .pdf z.B als Scan hat keine textblöcke (Bei eingescannten unterschriebenen Bewerbungen nicht unüblich) --> Methode, um zuverlässig Text in Bildern zu erkennen und zu extrahieren
##### 1 .py funktion die eine .pdf Datei annimmt, welche in 1.1.1 keine Textblöcke enthält (am besten Seite für seite prüfen. Also Seiten übergeben die keinen Text besitzen. Vllt wurde z.B. die letzte Seite unterschrieben, eingescannt und dann an die pdf angefügt). Am Ende alle pdf-seiten wieder verbinden. Gleicher return wie in 1.1.1. Am besten funktion von 1.1.2 in 1.1.1 aufrufen

#### Hier soll am Ende pro Dokument eine .txt Datei stehen (vllt auch was anderes je nachdem was besser für LLM ist)

### 2. Struktur schaffen

#### 2.1 RAG wäre hier gut. Also chunking mit overlap. Window z.B. 500 Zeichen mit 50 Zeichen overlap damit LLM kontext behält

#### 2.2 Jeder Chunk wird durch ein Embedding-Modell (z. B. OpenAI text-embedding-3 oder ein lokales HuggingFace-Modell) in Vektoren umgewandelt und in einer Vektordatenbank (wie ChromaDB, FAISS oder deiner bekannten PgVector) gespeichert.

### 3. LLM extrahiert aus der Struktur informationen

#### Erste Idee. Es ist ja quasi time series data (ergibt nur in der time series also im Wortfluss Sinn). Deswegen: Tokenisieren --> Attention --> ??? --> LLM

#### Dann Eingabe an LLM übergeben und Augabe ins Terminal oder auf Webseite schreiben

## Beispielfragen
### Wie lautet der Name des Bewerbers?
### Welche Fähigkeiten werden im Lebenslauf genannt?
### Wann hat der Bewerber seinen letzten Job begonnen?
#### Das System soll auf Grundlage der Dokumentinhalte passende und präzise Antworten auf die Fragen liefern.

## Kernanforderungen & Designziele
### Lösungsansatz und gewählte Architektur vollständig dokumentieren
### Architektur und Komponentenabläufe mit Diagrammen visualisieren
### Vollständige Nachvollziehbarkeit der Design- und Technologieentscheidungen
