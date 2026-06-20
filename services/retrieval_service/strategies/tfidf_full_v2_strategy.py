import pickle

import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity

from shared.config import TOP_K, ARTIFACTS_DIR
from services.document_store_service.document_database import get_document_by_id
from services.retrieval_service.strategies.base_strategy import RetrievalStrategy


TFIDF_FULL_V2_DIR = ARTIFACTS_DIR / "tfidf_full_v2"


class TfidfFullV2RetrievalStrategy(RetrievalStrategy):
    def __init__(self):
        self.vectorizer, self.tfidf_matrix, self.doc_ids = self._load_artifacts()

    def _load_artifacts(self):
        print("Loading TF-IDF Full v2 vectorizer...")
        with open(TFIDF_FULL_V2_DIR / "tfidf_vectorizer.pkl", "rb") as file:
            vectorizer = pickle.load(file)

        print("Loading TF-IDF sparse matrix...")
        tfidf_matrix = scipy.sparse.load_npz(
            str(TFIDF_FULL_V2_DIR / "tfidf_matrix.npz")
        )
        print(f"  Matrix shape : {tfidf_matrix.shape}")

        print("Loading doc IDs...")
        with open(TFIDF_FULL_V2_DIR / "tfidf_doc_ids.pkl", "rb") as file:
            doc_ids = pickle.load(file)

        print(f"TF-IDF Full v2 strategy ready. ({len(doc_ids):,} documents)")
        return vectorizer, tfidf_matrix, doc_ids

    def search(self, query: str, top_k: int = TOP_K):
        # الخطوة 1: تحويل الاستعلام إلى sparse vector
        # لاحظي: لا يوجد toarray() هنا — يبقى sparse
        query_vector = self.vectorizer.transform([query])

        # الخطوة 2: حساب Cosine Similarity بين الاستعلام وكل الوثائق
        # sklearn تدعم cosine_similarity على sparse مباشرة
        # النتيجة: مصفوفة شكلها (1, 522931) تحتوي درجة تشابه كل وثيقة
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # الخطوة 3: أخذ أعلى top_k درجة
        # argsort يرتّب من الأصغر للأكبر، لذلك نعكس بـ [::-1]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # الخطوة 4: بناء النتائج
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(similarities[idx])
            if score == 0.0:
                break  # لا داعي لإرجاع نتائج بدون أي تشابه
            doc_id = self.doc_ids[idx]
            results.append({
                "rank":   rank,
                "doc_id": doc_id,
                "score":  score,
                "text":   get_document_by_id(doc_id)
            })

        return results