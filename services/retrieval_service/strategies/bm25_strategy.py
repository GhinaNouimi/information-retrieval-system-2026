import pickle

import numpy as np

from shared.config import TOP_K, BM25_SAMPLE_DIR
from services.preprocessing_service.query_preprocessing import preprocess_query
from services.document_store_service.document_database import get_document_by_id
from services.retrieval_service.strategies.base_strategy import RetrievalStrategy


class BM25RetrievalStrategy(RetrievalStrategy):
    def __init__(self):
        self.bm25, self.doc_ids = self._load_artifacts()

    def _load_artifacts(self):
        with open(BM25_SAMPLE_DIR / "bm25_index.pkl", "rb") as file:
            bm25 = pickle.load(file)

        with open(BM25_SAMPLE_DIR / "bm25_doc_ids.pkl", "rb") as file:
            doc_ids = pickle.load(file)

        return bm25, doc_ids

    def search(self, query: str, top_k: int = TOP_K):
        query_tokens = preprocess_query(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for rank, index in enumerate(ranked_indices, start=1):
            doc_id = self.doc_ids[index]

            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": scores[index],
                "text": get_document_by_id(doc_id)
            })

        return results