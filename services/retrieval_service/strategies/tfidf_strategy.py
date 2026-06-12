import pickle

from sklearn.metrics.pairwise import cosine_similarity

from shared.config import TOP_K, TFIDF_SAMPLE_DIR
from services.document_store_service.document_database import get_document_by_id
from services.retrieval_service.strategies.base_strategy import RetrievalStrategy


class TfidfRetrievalStrategy(RetrievalStrategy):
    def __init__(self):
        self.vectorizer, self.tfidf_matrix, self.doc_ids = self._load_artifacts()

    def _load_artifacts(self):
        with open(TFIDF_SAMPLE_DIR / "tfidf_vectorizer.pkl", "rb") as file:
            vectorizer = pickle.load(file)

        with open(TFIDF_SAMPLE_DIR / "tfidf_matrix.pkl", "rb") as file:
            tfidf_matrix = pickle.load(file)

        with open(TFIDF_SAMPLE_DIR / "doc_ids.pkl", "rb") as file:
            doc_ids = pickle.load(file)

        return vectorizer, tfidf_matrix, doc_ids

    def search(self, query: str, top_k: int = TOP_K):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        ranked_indices = scores.argsort()[::-1][:top_k]

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