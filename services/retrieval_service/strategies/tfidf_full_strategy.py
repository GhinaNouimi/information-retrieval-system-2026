import pickle

import numpy as np
import faiss

from shared.config import TOP_K, ARTIFACTS_DIR
from services.document_store_service.document_database import get_document_by_id
from services.retrieval_service.strategies.base_strategy import RetrievalStrategy


TFIDF_FULL_DIR = ARTIFACTS_DIR / "tfidf_full"


class TfidfFullRetrievalStrategy(RetrievalStrategy):
    def __init__(self):
        self.vectorizer, self.index, self.doc_ids = self._load_artifacts()

    def _load_artifacts(self):
        print("Loading TF-IDF Full vectorizer...")
        with open(TFIDF_FULL_DIR / "tfidf_vectorizer.pkl", "rb") as file:
            vectorizer = pickle.load(file)

        print("Loading FAISS index...")
        index = faiss.read_index(str(TFIDF_FULL_DIR / "faiss_index.bin"))
        index.nprobe = 10

        with open(TFIDF_FULL_DIR / "tfidf_doc_ids.pkl", "rb") as file:
            doc_ids = pickle.load(file)

        print(f"TF-IDF Full strategy ready. ({index.ntotal:,} vectors)")
        return vectorizer, index, doc_ids

    def search(self, query: str, top_k: int = TOP_K):
        query_vector = self.vectorizer.transform([query])
        query_dense  = query_vector.toarray().astype(np.float32)
        faiss.normalize_L2(query_dense)

        scores, indices = self.index.search(query_dense, top_k)

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