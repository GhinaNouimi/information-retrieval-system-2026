import math


def precision_at_k(retrieved_doc_ids: list, relevant_doc_ids: set, k: int = 10) -> float:
    """
    Precision@K:
    من أول K نتيجة، كم منها صحيحة؟

    مثال:
    - retrieved = [doc1, doc2, doc3, doc4, doc5]
    - relevant  = {doc1, doc3}
    - k = 5
    - النتيجة = 2/5 = 0.4
    """
    top_k = retrieved_doc_ids[:k]

    if len(top_k) == 0:
        return 0.0

    hits = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)

    return hits / len(top_k)


def recall(retrieved_doc_ids: list, relevant_doc_ids: set) -> float:
    """
    Recall:
    من كل الوثائق الصحيحة الموجودة، كم وجدنا؟

    مثال:
    - retrieved = [doc1, doc2, doc3]
    - relevant  = {doc1, doc3, doc5, doc7}
    - النتيجة = 2/4 = 0.5
    """
    if len(relevant_doc_ids) == 0:
        return 0.0

    hits = sum(1 for doc_id in retrieved_doc_ids if doc_id in relevant_doc_ids)

    return hits / len(relevant_doc_ids)


def average_precision(retrieved_doc_ids: list, relevant_doc_ids: set) -> float:
    """
    Average Precision (AP):
    يُستخدم لحساب MAP لاحقاً.

    يحسب دقة النظام عند كل نقطة يجد فيها وثيقة صحيحة.

    مثال:
    - retrieved = [doc1, doc2, doc3, doc4, doc5]
    - relevant  = {doc1, doc3}
    - عند doc1 (rank 1): precision = 1/1 = 1.0
    - عند doc3 (rank 3): precision = 2/3 = 0.667
    - AP = (1.0 + 0.667) / 2 = 0.833
    """
    if len(relevant_doc_ids) == 0:
        return 0.0

    hits = 0
    sum_precision = 0.0

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            hits += 1
            sum_precision += hits / rank

    if hits == 0:
        return 0.0

    return sum_precision / len(relevant_doc_ids)


def ndcg_at_k(retrieved_doc_ids: list, relevant_doc_ids: set, k: int = 10) -> float:
    """
    nDCG@K (Normalized Discounted Cumulative Gain):
    هل الوثائق الصحيحة جاءت في أعلى القائمة؟

    يعطي وزناً أكبر للوثائق الصحيحة التي تظهر في مراتب أعلى.

    مثال:
    - retrieved = [doc1, doc2, doc3]
    - relevant  = {doc1, doc3}
    - DCG  = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
    - IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.63 = 1.63
    - nDCG = 1.5 / 1.63 = 0.92
    """
    top_k = retrieved_doc_ids[:k]

    if len(top_k) == 0 or len(relevant_doc_ids) == 0:
        return 0.0

    # حساب DCG الفعلي
    dcg = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_doc_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # حساب IDCG (أفضل ترتيب ممكن)
    ideal_hits = min(len(relevant_doc_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg