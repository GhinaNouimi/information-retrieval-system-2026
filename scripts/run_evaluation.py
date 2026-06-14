from services.retrieval_service.strategies.tfidf_strategy import TfidfRetrievalStrategy
from services.retrieval_service.strategies.bm25_strategy import BM25RetrievalStrategy
from services.evaluation_service.evaluator import evaluate


def print_results(model_name: str, results: dict):
    print("=" * 40)
    print(f"Model: {model_name}")
    print("=" * 40)
    print(f"Queries Evaluated : {results['num_queries_evaluated']}")
    print(f"Precision@10      : {results['precision_at_k']:.4f}")
    print(f"Recall            : {results['recall']:.4f}")
    print(f"MAP               : {results['map']:.4f}")
    print(f"nDCG@10           : {results['ndcg_at_k']:.4f}")
    print()


def main():
    print("Loading TF-IDF strategy...")
    tfidf_strategy = TfidfRetrievalStrategy()

    print("Running TF-IDF evaluation...")
    tfidf_results = evaluate(tfidf_strategy)
    print_results("TF-IDF", tfidf_results)

    print("Loading BM25 strategy...")
    bm25_strategy = BM25RetrievalStrategy()

    print("Running BM25 evaluation...")
    bm25_results = evaluate(bm25_strategy)
    print_results("BM25", bm25_results)

    print("=" * 40)
    print("Comparison Summary")
    print("=" * 40)
    print(f"{'Metric':<20} {'TF-IDF':>10} {'BM25':>10}")
    print("-" * 40)
    print(f"{'Precision@10':<20} {tfidf_results['precision_at_k']:>10.4f} {bm25_results['precision_at_k']:>10.4f}")
    print(f"{'Recall':<20} {tfidf_results['recall']:>10.4f} {bm25_results['recall']:>10.4f}")
    print(f"{'MAP':<20} {tfidf_results['map']:>10.4f} {bm25_results['map']:>10.4f}")
    print(f"{'nDCG@10':<20} {tfidf_results['ndcg_at_k']:>10.4f} {bm25_results['ndcg_at_k']:>10.4f}")


if __name__ == "__main__":
    main()