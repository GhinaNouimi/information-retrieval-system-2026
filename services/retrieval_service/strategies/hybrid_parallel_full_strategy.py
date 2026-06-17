from shared.config import TOP_K
from services.retrieval_service.strategies.base_strategy import RetrievalStrategy
from services.retrieval_service.strategies.bm25_full_strategy import BM25FullRetrievalStrategy
from services.retrieval_service.strategies.embedding_full_strategy import EmbeddingFullRetrievalStrategy
from services.document_store_service.document_database import get_document_by_id


CANDIDATE_SIZE = 100
RRF_K          = 60   # القيمة المعتمدة في الأبحاث


class HybridParallelFullRetrievalStrategy(RetrievalStrategy):
    """
    Hybrid Parallel Full على كامل الـ 522,931 وثيقة:

    BM25 Full و Embedding Full يعملان بشكل مستقل،
    ثم تُدمج نتائجهما باستخدام Reciprocal Rank Fusion (RRF).

    طريقة RRF:
    لكل وثيقة: score = 1/(60 + rank_bm25) + 1/(60 + rank_embedding)
    الوثيقة التي تظهر في مراتب عالية في كلا النموذجين تحصل على أعلى درجة.
    """

    def __init__(self):
        print("Loading BM25 Full strategy...")
        self.bm25_strategy = BM25FullRetrievalStrategy()

        print("Loading Embedding Full strategy...")
        self.embedding_strategy = EmbeddingFullRetrievalStrategy()

        print("Hybrid Parallel Full strategy ready.")

    def search(self, query: str, top_k: int = TOP_K):

        # الخطوة 1: كلا النموذجين يبحثان بشكل مستقل على 522K وثيقة
        bm25_results      = self.bm25_strategy.search(query,      top_k=CANDIDATE_SIZE)
        embedding_results = self.embedding_strategy.search(query, top_k=CANDIDATE_SIZE)

        # الخطوة 2: حساب درجة RRF لكل وثيقة
        rrf_scores = {}

        for result in bm25_results:
            doc_id = result["doc_id"]
            rank   = result["rank"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank)

        for result in embedding_results:
            doc_id = result["doc_id"]
            rank   = result["rank"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank)

        # الخطوة 3: ترتيب الوثائق حسب درجة RRF تنازلياً
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # الخطوة 4: أخذ أفضل top_k نتيجة
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], start=1):
            results.append({
                "rank":   rank,
                "doc_id": doc_id,
                "score":  score,
                "text":   get_document_by_id(doc_id),
            })

        return results