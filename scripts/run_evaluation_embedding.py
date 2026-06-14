from services.retrieval_service.strategies.bm25_large_strategy import BM25LargeRetrievalStrategy
from services.retrieval_service.strategies.embedding_strategy import EmbeddingRetrievalStrategy
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
    print("Loading BM25 Large (50,000 docs)...")
    bm25_strategy = BM25LargeRetrievalStrategy()

    print("Running BM25 evaluation...")
    bm25_results = evaluate(bm25_strategy)
    print_results("BM25 - 50,000 docs", bm25_results)

    print("Loading Embedding (50,000 docs)...")
    embedding_strategy = EmbeddingRetrievalStrategy()

    print("Running Embedding evaluation...")
    embedding_results = evaluate(embedding_strategy)
    print_results("Embedding - 50,000 docs", embedding_results)

    print("=" * 50)
    print("Comparison: BM25 vs Embedding")
    print("=" * 50)
    print(f"{'Metric':<20} {'BM25':>10} {'Embedding':>12} {'Winner':>10}")
    print("-" * 55)

    metrics = [
        ("Precision@10", "precision_at_k"),
        ("Recall",       "recall"),
        ("MAP",          "map"),
        ("nDCG@10",      "ndcg_at_k"),
    ]

    for label, key in metrics:
        bm25_val      = bm25_results[key]
        embedding_val = embedding_results[key]
        winner = "BM25" if bm25_val > embedding_val else "Embedding"
        print(f"{label:<20} {bm25_val:>10.4f} {embedding_val:>12.4f} {winner:>10}")


if __name__ == "__main__":
    main()