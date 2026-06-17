import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from shared.config import TOP_K, ARTIFACTS_DIR
from services.retrieval_service.strategies.base_strategy import RetrievalStrategy
from services.retrieval_service.strategies.bm25_full_strategy import BM25FullRetrievalStrategy
from services.document_store_service.document_database import get_document_by_id


MODEL_NAME          = "all-MiniLM-L6-v2"
BM25_CANDIDATE_SIZE = 100


class HybridSerialFullV2RetrievalStrategy(RetrievalStrategy):
    """
    Hybrid Serial Full على كامل الـ 522,931 وثيقة:

    الخطوة 1: BM25 Full يسترجع أفضل 100 مرشح من 522,931 وثيقة
    الخطوة 2: Embedding يعيد ترتيب الـ 100 فقط (وليس كل الـ 522K)
    الخطوة 3: نأخذ أفضل top_k من النتائج المعاد ترتيبها

    لماذا هذا ذكي؟
    - BM25 سريع جداً على 522K وثيقة
    - Embedding يعمل على 100 وثيقة فقط وليس 522K
    - النتيجة: دقة Embedding مع سرعة BM25
    """

    def __init__(self):
        print("Loading BM25 Full strategy...")
        self.bm25_strategy = BM25FullRetrievalStrategy()

        print("Loading Sentence Transformer model...")
        self.model = SentenceTransformer(MODEL_NAME)

        print("Hybrid Serial Full v2 strategy ready.")

    def search(self, query: str, top_k: int = TOP_K):

        # الخطوة 1: BM25 Full يسترجع 100 مرشح من 522K وثيقة
        bm25_candidates = self.bm25_strategy.search(query, top_k=BM25_CANDIDATE_SIZE)

        if not bm25_candidates:
            return []

        candidate_texts   = [c["text"] or "" for c in bm25_candidates]
        candidate_doc_ids = [c["doc_id"] for c in bm25_candidates]

        # الخطوة 2: Embedding يعيد ترتيب الـ 100 مرشح فقط
        query_embedding      = self.model.encode([query],          convert_to_numpy=True).astype(np.float32)
        candidate_embeddings = self.model.encode(candidate_texts,  convert_to_numpy=True).astype(np.float32)

        # normalize ثم dot product = cosine similarity
        faiss.normalize_L2(query_embedding)
        faiss.normalize_L2(candidate_embeddings)

        scores = (query_embedding @ candidate_embeddings.T).flatten()

        # الخطوة 3: ترتيب المرشحين حسب درجة Embedding وأخذ أفضل top_k
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            doc_id = candidate_doc_ids[idx]
            results.append({
                "rank":   rank,
                "doc_id": doc_id,
                "score":  float(scores[idx]),
                "text":   get_document_by_id(doc_id),
            })

        return results