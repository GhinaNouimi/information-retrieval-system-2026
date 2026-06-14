from shared.config import TOP_K, ARTIFACTS_DIR
from services.retrieval_service.strategies.base_strategy import RetrievalStrategy
from services.retrieval_service.strategies.bm25_large_strategy import BM25LargeRetrievalStrategy
from services.retrieval_service.strategies.embedding_strategy import EmbeddingRetrievalStrategy
from services.document_store_service.document_database import get_document_by_id


# عدد النتائج التي يسترجعها كل نموذج قبل الدمج
CANDIDATE_SIZE = 100

# ثابت RRF — القيمة 60 هي القيمة المعتمدة في الأبحاث
RRF_K = 60


class HybridParallelRetrievalStrategy(RetrievalStrategy):
    """
    Hybrid Parallel (المتوازي):

    BM25 و Embedding يعملان في نفس الوقت بشكل مستقل،
    ثم تُدمج نتائجهما باستخدام Reciprocal Rank Fusion (RRF).

    طريقة RRF:
    لكل وثيقة: score = 1/(k + rank_bm25) + 1/(k + rank_embedding)
    الوثيقة التي تظهر في مراتب عالية في كلا النموذجين تحصل على أعلى درجة.

    مثال:
    - وثيقة A: rank 1 في BM25، rank 3 في Embedding
      score = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
    - وثيقة B: rank 1 في Embedding فقط، غير موجودة في BM25
      score = 1/(60+1) = 0.0164
    وثيقة A تفوز لأنها قوية في كلا النموذجين.
    """

    def __init__(self):
        self.bm25_strategy = BM25LargeRetrievalStrategy()
        self.embedding_strategy = EmbeddingRetrievalStrategy()

    def search(self, query: str, top_k: int = TOP_K):

        # الخطوة 1: كلا النموذجين يبحثان بشكل مستقل
        bm25_results      = self.bm25_strategy.search(query, top_k=CANDIDATE_SIZE)
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

        # الخطوة 3: ترتيب الوثائق حسب درجة RRF
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # الخطوة 4: أخذ أفضل top_k نتيجة
        results = []

        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], start=1):
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": score,
                "text": get_document_by_id(doc_id)
            })

        return results