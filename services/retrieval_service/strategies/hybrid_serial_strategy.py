import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from shared.config import TOP_K, ARTIFACTS_DIR
from services.retrieval_service.strategies.base_strategy import RetrievalStrategy
from services.retrieval_service.strategies.bm25_large_strategy import BM25LargeRetrievalStrategy
from services.document_store_service.document_database import get_document_by_id


MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_LARGE_DIR = ARTIFACTS_DIR / "embedding_large"

# عدد النتائج التي يسترجعها BM25 قبل أن يعيد Embedding ترتيبها
BM25_CANDIDATE_SIZE = 100


class HybridSerialRetrievalStrategy(RetrievalStrategy):
    """
    Hybrid Serial (التسلسلي):

    الخطوة 1: BM25 يسترجع أفضل 100 وثيقة مرشحة
    الخطوة 2: Embedding يعيد ترتيب هذه الـ 100 وثيقة فقط
    الخطوة 3: نأخذ أفضل top_k من النتائج المعاد ترتيبها

    هذا أسرع من Embedding الكامل لأنه لا يحسب التشابه
    مع كل الوثائق بل مع 100 فقط.
    """

    def __init__(self):
        self.bm25_strategy = BM25LargeRetrievalStrategy()
        self.model = SentenceTransformer(MODEL_NAME)

    def search(self, query: str, top_k: int = TOP_K):

        # الخطوة 1: BM25 يسترجع أفضل 100 مرشح
        bm25_candidates = self.bm25_strategy.search(query, top_k=BM25_CANDIDATE_SIZE)

        if not bm25_candidates:
            return []

        # الخطوة 2: Embedding يعيد ترتيب المرشحين
        candidate_texts = [c["text"] or "" for c in bm25_candidates]
        candidate_doc_ids = [c["doc_id"] for c in bm25_candidates]

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        candidate_embeddings = self.model.encode(candidate_texts, convert_to_numpy=True)

        scores = cosine_similarity(query_embedding, candidate_embeddings).flatten()

        # الخطوة 3: ترتيب المرشحين حسب درجة Embedding
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for rank, index in enumerate(ranked_indices, start=1):
            doc_id = candidate_doc_ids[index]

            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": float(scores[index]),
                "text": get_document_by_id(doc_id)
            })

        return results