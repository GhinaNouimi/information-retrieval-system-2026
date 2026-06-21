import math


def precision_at_k(retrieved_doc_ids: list, relevant_doc_ids: set, k: int = 10) -> float:
   
    top_k = retrieved_doc_ids[:k]

    if len(top_k) == 0:
        return 0.0

    hits = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)

    return hits / len(top_k)


def recall(retrieved_doc_ids: list, relevant_doc_ids: set) -> float:

    if len(relevant_doc_ids) == 0:
        return 0.0

    hits = sum(1 for doc_id in retrieved_doc_ids if doc_id in relevant_doc_ids)

    return hits / len(relevant_doc_ids)


def average_precision(retrieved_doc_ids: list, relevant_doc_ids: set) -> float:
    
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
    
    top_k = retrieved_doc_ids[:k]

    if len(top_k) == 0 or len(relevant_doc_ids) == 0:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_doc_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_doc_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg