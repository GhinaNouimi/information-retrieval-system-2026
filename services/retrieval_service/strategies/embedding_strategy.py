import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from shared.config import TOP_K, ARTIFACTS_DIR
from services.document_store_service.document_database import get_document_by_id
from services.retrieval_service.strategies.base_strategy import RetrievalStrategy


EMBEDDING_LARGE_DIR = ARTIFACTS_DIR / "embedding_large"
MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingRetrievalStrategy(RetrievalStrategy):
    def __init__(self):
        self.model, self.embeddings_matrix, self.doc_ids = self._load_artifacts()

    def _load_artifacts(self):
        print("Loading embedding model...")
        model = SentenceTransformer(MODEL_NAME)

        print("Loading embeddings matrix...")
        embeddings_matrix = np.load(EMBEDDING_LARGE_DIR / "embeddings_matrix.npy")

        with open(EMBEDDING_LARGE_DIR / "embedding_doc_ids.pkl", "rb") as file:
            doc_ids = pickle.load(file)

        print("Embedding strategy ready.")
        return model, embeddings_matrix, doc_ids

    def search(self, query: str, top_k: int = TOP_K):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        scores = cosine_similarity(query_embedding, self.embeddings_matrix).flatten()
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for rank, index in enumerate(ranked_indices, start=1):
            doc_id = self.doc_ids[index]

            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": float(scores[index]),
                "text": get_document_by_id(doc_id)
            })

        return results