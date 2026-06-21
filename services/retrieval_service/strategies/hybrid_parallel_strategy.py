from shared.config import TOP_K, ARTIFACTS_DIR
from services.retrieval_service.strategies.base_strategy import RetrievalStrategy
from services.retrieval_service.strategies.bm25_large_strategy import BM25LargeRetrievalStrategy
from services.retrieval_service.strategies.embedding_strategy import EmbeddingRetrievalStrategy
from services.document_store_service.document_database import get_document_by_id


CANDIDATE_SIZE = 100

RRF_K = 60


class HybridParallelRetrievalStrategy(RetrievalStrategy):
   

    def __init__(self):
        self.bm25_strategy = BM25LargeRetrievalStrategy()
        self.embedding_strategy = EmbeddingRetrievalStrategy()

    def search(self, query: str, top_k: int = TOP_K):
        bm25_results      = self.bm25_strategy.search(query, top_k=CANDIDATE_SIZE)
        embedding_results = self.embedding_strategy.search(query, top_k=CANDIDATE_SIZE)

        rrf_scores = {}

        for result in bm25_results:
            doc_id = result["doc_id"]
            rank   = result["rank"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank)

        for result in embedding_results:
            doc_id = result["doc_id"]
            rank   = result["rank"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []

        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], start=1):
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": score,
                "text": get_document_by_id(doc_id)
            })

        return results