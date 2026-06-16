import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from shared.config import TOP_K, ARTIFACTS_DIR
from services.retrieval_service.strategies.base_strategy import RetrievalStrategy
from services.retrieval_service.strategies.bm25_full_strategy import BM25FullRetrievalStrategy
from services.document_store_service.document_database import get_document_by_id


EMBEDDING_FULL_DIR = ARTIFACTS_DIR / "embedding_full"
MODEL_NAME         = "all-MiniLM-L6-v2"
BM25_CANDIDATE_SIZE = 100


class HybridSerialFullRetrievalStrategy(RetrievalStrategy):
    """
    Hybrid Serial على كامل الـ Dataset:
    - BM25 Full يسترجع أفضل 100 مرشح من 522,931 وثيقة
    - Embedding يعيد ترتيب الـ 100 فقط
    - نأخذ أفضل top_k
    """

    def __init__(self):
        self.bm25_strategy = BM25FullRetrievalStrategy()
        self.model         = SentenceTransformer(MODEL_NAME)

    def search(self, query: str, top_k: int = TOP_K):

        # الخطوة 1: BM25 Full يسترجع 100 مرشح
        bm25_candidates    = self.bm25_strategy.search(query, top_k=BM25_CANDIDATE_SIZE)

        if not bm25_candidates:
            return []

        candidate_texts   = [c["text"] or "" for c in bm25_candidates]
        candidate_doc_ids = [c["doc_id"] for c in bm25_candidates]

        # الخطوة 2: Embedding يعيد ترتيب المرشحين
        query_embedding      = self.model.encode([query], convert_to_numpy=True)
        candidate_embeddings = self.model.encode(candidate_texts, convert_to_numpy=True)

        faiss.normalize_L2(query_embedding)
        faiss.normalize_L2(candidate_embeddings)

        scores = (query_embedding @ candidate_embeddings.T).flatten()

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for rank, index in enumerate(ranked_indices, start=1):
            doc_id = candidate_doc_ids[index]

            results.append({
                "rank":   rank,
                "doc_id": doc_id,
                "score":  float(scores[index]),
                "text":   get_document_by_id(doc_id)
            })

        return results