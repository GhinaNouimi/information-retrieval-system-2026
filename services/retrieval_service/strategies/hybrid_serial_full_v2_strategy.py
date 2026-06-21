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
  

    def __init__(self):
        print("Loading BM25 Full strategy...")
        self.bm25_strategy = BM25FullRetrievalStrategy()

        print("Loading Sentence Transformer model...")
        self.model = SentenceTransformer(MODEL_NAME)

        print("Hybrid Serial Full v2 strategy ready.")

    def search(self, query: str, top_k: int = TOP_K):

        bm25_candidates = self.bm25_strategy.search(query, top_k=BM25_CANDIDATE_SIZE)

        if not bm25_candidates:
            return []

        candidate_texts   = [c["text"] or "" for c in bm25_candidates]
        candidate_doc_ids = [c["doc_id"] for c in bm25_candidates]

        query_embedding      = self.model.encode([query],          convert_to_numpy=True).astype(np.float32)
        candidate_embeddings = self.model.encode(candidate_texts,  convert_to_numpy=True).astype(np.float32)

        faiss.normalize_L2(query_embedding)
        faiss.normalize_L2(candidate_embeddings)

        scores = (query_embedding @ candidate_embeddings.T).flatten()

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