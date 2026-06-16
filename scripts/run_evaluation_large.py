from services.retrieval_service.strategies.bm25_strategy import BM25RetrievalStrategy
from services.retrieval_service.strategies.bm25_large_strategy import BM25LargeRetrievalStrategy
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
    print("Loading BM25 Small (10,000 docs)...")
    bm25_small = BM25RetrievalStrategy()

    print("Running evaluation on BM25 Small...")
    small_results = evaluate(bm25_small)
    print_results("BM25 - 10,000 docs", small_results)

    print("Loading BM25 Large (50,000 docs)...")
    bm25_large = BM25LargeRetrievalStrategy()

    print("Running evaluation on BM25 Large...")
    large_results = evaluate(bm25_large)
    print_results("BM25 - 50,000 docs", large_results)

    print("=" * 40)
    print("Comparison: Small vs Large")
    print("=" * 40)
    print(f"{'Metric':<20} {'10K docs':>10} {'50K docs':>10} {'Change':>10}")
    print("-" * 50)

    metrics = [
        ("Precision@10", "precision_at_k"),
        ("Recall",       "recall"),
        ("MAP",          "map"),
        ("nDCG@10",      "ndcg_at_k"),
    ]

    for label, key in metrics:
        small_val = small_results[key]
        large_val = large_results[key]
        change = ((large_val - small_val) / small_val * 100) if small_val > 0 else 0
        print(f"{label:<20} {small_val:>10.4f} {large_val:>10.4f} {change:>+9.1f}%")


if __name__ == "__main__":
    main()