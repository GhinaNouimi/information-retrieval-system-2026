import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from shared.config import TOP_K, ARTIFACTS_DIR
from services.document_store_service.document_database import get_document_by_id
from services.retrieval_service.strategies.base_strategy import RetrievalStrategy


EMBEDDING_FULL_DIR = ARTIFACTS_DIR / "embedding_full"
MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingFullRetrievalStrategy(RetrievalStrategy):
    """
    Embedding Retrieval على كامل الـ Dataset (522,931 وثيقة)
    باستخدام FAISS للبحث السريع.
    """

    def __init__(self):
        self.model, self.index, self.doc_ids = self._load_artifacts()

    def _load_artifacts(self):
        print("Loading embedding model...")
        model = SentenceTransformer(MODEL_NAME)

        print("Loading FAISS index...")
        index = faiss.read_index(str(EMBEDDING_FULL_DIR / "faiss_index.bin"))

        with open(EMBEDDING_FULL_DIR / "embedding_doc_ids.pkl", "rb") as file:
            doc_ids = pickle.load(file)

        print(f"Embedding Full strategy ready. ({index.ntotal:,} vectors)")
        return model, index, doc_ids

    def search(self, query: str, top_k: int = TOP_K):
        # تحويل الـ Query إلى متجه
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # تطبيع المتجه (ضروري لـ FAISS مع Cosine Similarity)
        faiss.normalize_L2(query_embedding)

        # البحث في FAISS
        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:
                continue

            doc_id = self.doc_ids[idx]

            results.append({
                "rank":   rank,
                "doc_id": doc_id,
                "score":  float(score),
                "text":   get_document_by_id(doc_id)
            })

        return results