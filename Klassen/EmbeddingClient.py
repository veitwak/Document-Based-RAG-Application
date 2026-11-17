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