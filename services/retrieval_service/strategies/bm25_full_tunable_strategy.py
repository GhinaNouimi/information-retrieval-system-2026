import pickle

import numpy as np
from rank_bm25 import BM25Okapi

from shared.config import TOP_K, ARTIFACTS_DIR
from services.document_store_service.document_database import get_document_by_id
from services.retrieval_service.strategies.base_strategy import RetrievalStrategy


BM25_FULL_DIR = ARTIFACTS_DIR / "bm25_full"

DEFAULT_K1 = 1.5
DEFAULT_B  = 0.75


class BM25FullTunableStrategy(RetrievalStrategy):
    """
    BM25 Full مع إمكانية تغيير المعاملات k1 و b من الواجهة.

    k1: يتحكم في تأثير تكرار الكلمة في الوثيقة
        - قيمة منخفضة (0.5): تكرار الكلمة لا يؤثر كثيراً
        - قيمة عالية (2.5): تكرار الكلمة يرفع الدرجة أكثر
        - القيمة الموصى بها: 1.5

    b:  يتحكم في تأثير طول الوثيقة
        - b=0.0: طول الوثيقة لا يؤثر على الدرجة
        - b=1.0: تطبيع كامل حسب طول الوثيقة
        - القيمة الموصى بها: 0.75
    """

    def __init__(self):
        self.corpus_tokens, self.doc_ids = self._load_artifacts()
        self._k1   = DEFAULT_K1
        self._b    = DEFAULT_B
        print(f"Building BM25 with default parameters (k1={self._k1}, b={self._b})...")
        self._bm25 = BM25Okapi(self.corpus_tokens, k1=self._k1, b=self._b)
        print("BM25 Full Tunable strategy ready.")

    def _load_artifacts(self):
        print("Loading BM25 corpus tokens...")
        with open(BM25_FULL_DIR / "bm25_corpus_tokens.pkl", "rb") as file:
            corpus_tokens = pickle.load(file)
        with open(BM25_FULL_DIR / "bm25_doc_ids.pkl", "rb") as file:
            doc_ids = pickle.load(file)
        print(f"Corpus loaded. ({len(doc_ids):,} documents)")
        return corpus_tokens, doc_ids

    def _rebuild_if_needed(self, k1: float, b: float):
        """يعيد بناء BM25 فقط إذا تغيرت المعاملات."""
        if k1 != self._k1 or b != self._b:
            print(f"Rebuilding BM25 with k1={k1}, b={b}...")
            self._bm25 = BM25Okapi(self.corpus_tokens, k1=k1, b=b)
            self._k1   = k1
            self._b    = b
            print("Rebuild complete.")

    def search(self, query_tokens: list, top_k: int = TOP_K):
        """البحث بالمعاملات الحالية."""
        return self.search_with_params(query_tokens, top_k=top_k,
                                       k1=self._k1, b=self._b)

    def search_with_params(self, query_tokens: list, top_k: int = TOP_K,
                           k1: float = DEFAULT_K1, b: float = DEFAULT_B):
        """البحث مع تحديد المعاملات مباشرة من الواجهة."""
        self._rebuild_if_needed(k1, b)
        scores         = self._bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            doc_id = self.doc_ids[idx]
            results.append({
                "rank":   rank,
                "doc_id": doc_id,
                "score":  float(scores[idx]),
                "text":   get_document_by_id(doc_id),
            })
        return results