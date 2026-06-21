import ir_datasets

from shared.config import DATASET_ID, TOP_K
from services.evaluation_service.metrics import (
    precision_at_k,
    recall,
    average_precision,
    ndcg_at_k,
)


def load_qrels() -> dict:
   
    dataset = ir_datasets.load(DATASET_ID)

    qrels = {}

    for qrel in dataset.qrels_iter():
        if qrel.relevance >= 1:
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = set()
            qrels[qrel.query_id].add(qrel.doc_id)

    return qrels


def load_queries() -> dict:
   
    dataset = ir_datasets.load(DATASET_ID)

    queries = {}

    for query in dataset.queries_iter():
        queries[query.query_id] = query.text

    return queries


def evaluate(strategy, top_k: int = TOP_K) -> dict:
   
    print("Loading queries and qrels from dataset...")

    qrels = load_qrels()
    queries = load_queries()

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

        search_results = strategy.search(query_text, top_k=top_k)
        retrieved_doc_ids = [result["doc_id"] for result in search_results]

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