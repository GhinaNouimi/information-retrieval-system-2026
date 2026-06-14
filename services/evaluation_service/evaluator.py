import ir_datasets

from shared.config import DATASET_ID, TOP_K
from services.evaluation_service.metrics import (
    precision_at_k,
    recall,
    average_precision,
    ndcg_at_k,
)


def load_qrels() -> dict:
    """
    يقرأ Qrels من Dataset ويُرجعها كـ Dictionary.

    الشكل الناتج:
    {
        "query_id_1": {"doc_id_a", "doc_id_b"},
        "query_id_2": {"doc_id_c"},
        ...
    }

    relevance >= 1 تعني أن الوثيقة ذات صلة.
    """
    dataset = ir_datasets.load(DATASET_ID)

    qrels = {}

    for qrel in dataset.qrels_iter():
        if qrel.relevance >= 1:
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = set()
            qrels[qrel.query_id].add(qrel.doc_id)

    return qrels


def load_queries() -> dict:
    """
    يقرأ الـ Queries من Dataset ويُرجعها كـ Dictionary.

    الشكل الناتج:
    {
        "query_id_1": "what is machine learning",
        "query_id_2": "how to invest in stocks",
        ...
    }
    """
    dataset = ir_datasets.load(DATASET_ID)

    queries = {}

    for query in dataset.queries_iter():
        queries[query.query_id] = query.text

    return queries


def evaluate(strategy, top_k: int = TOP_K) -> dict:
    """
    يُشغّل التقييم الكامل على Strategy معينة.

    الخطوات:
    1. يقرأ الـ Queries والـ Qrels من Dataset
    2. لكل Query تملك Qrels يُشغّل البحث
    3. يحسب المقاييس الأربعة لكل Query
    4. يُرجع متوسط كل مقياس

    المعامل strategy:
    أي كائن يملك دالة search(query, top_k) وتُرجع
    قائمة من dictionaries تحتوي على مفتاح "doc_id".

    مثال:
    strategy = TfidfRetrievalStrategy()
    results  = evaluate(strategy)
    """
    print("Loading queries and qrels from dataset...")

    qrels = load_qrels()
    queries = load_queries()

    # نأخذ فقط الـ Queries التي لها Qrels
    evaluated_query_ids = [
        query_id
        for query_id in queries
        if query_id in qrels
    ]

    print(f"Total queries with qrels: {len(evaluated_query_ids)}")
    print(f"Running evaluation with top_k = {top_k}...")
    print()

    all_precision = []
    all_recall = []
    all_ap = []
    all_ndcg = []

    for i, query_id in enumerate(evaluated_query_ids, start=1):
        query_text = queries[query_id]
        relevant_docs = qrels[query_id]

        # تشغيل البحث
        search_results = strategy.search(query_text, top_k=top_k)
        retrieved_doc_ids = [result["doc_id"] for result in search_results]

        # حساب المقاييس
        p_at_k = precision_at_k(retrieved_doc_ids, relevant_docs, k=top_k)
        r = recall(retrieved_doc_ids, relevant_docs)
        ap = average_precision(retrieved_doc_ids, relevant_docs)
        ndcg = ndcg_at_k(retrieved_doc_ids, relevant_docs, k=top_k)

        all_precision.append(p_at_k)
        all_recall.append(r)
        all_ap.append(ap)
        all_ndcg.append(ndcg)

        if i % 100 == 0:
            print(f"Evaluated {i} / {len(evaluated_query_ids)} queries...")

    print()
    print(f"Evaluation complete. Total queries evaluated: {len(all_precision)}")

    results = {
        "num_queries_evaluated": len(all_precision),
        "precision_at_k": sum(all_precision) / len(all_precision),
        "recall": sum(all_recall) / len(all_recall),
        "map": sum(all_ap) / len(all_ap),
        "ndcg_at_k": sum(all_ndcg) / len(all_ndcg),
    }

    return results