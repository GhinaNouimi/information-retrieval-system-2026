from services.retrieval_service.strategies.bm25_large_strategy import BM25LargeRetrievalStrategy
from services.retrieval_service.strategies.bm25_full_strategy import BM25FullRetrievalStrategy
from services.retrieval_service.strategies.embedding_full_strategy import EmbeddingFullRetrievalStrategy
from services.evaluation_service.evaluator import evaluate


def print_results(model_name: str, results: dict):
    print("=" * 50)
    print(f"Model: {model_name}")
    print("=" * 50)
    print(f"Queries Evaluated : {results['num_queries_evaluated']:,}")
    print(f"Precision@10      : {results['precision_at_k']:.4f}")
    print(f"Recall            : {results['recall']:.4f}")
    print(f"MAP               : {results['map']:.4f}")
    print(f"nDCG@10           : {results['ndcg_at_k']:.4f}")
    print()


def main():
    print("[1/3] Loading BM25 Large (50K)...")
    bm25_50k = BM25LargeRetrievalStrategy()

    print("[2/3] Loading BM25 Full (522K)...")
    bm25_full = BM25FullRetrievalStrategy()

    print("[3/3] Loading Embedding Full (522K)...")
    emb_full = EmbeddingFullRetrievalStrategy()

    print()
    print("Starting evaluation...")
    print()

    print("Evaluating BM25 50K...")
    bm25_50k_results = evaluate(bm25_50k)
    print_results("BM25 - 50,000 docs", bm25_50k_results)

    print("Evaluating BM25 Full...")
    bm25_full_results = evaluate(bm25_full)
    print_results("BM25 - 522,931 docs", bm25_full_results)

    print("Evaluating Embedding Full...")
    emb_full_results = evaluate(emb_full)
    print_results("Embedding Full - 522,931 docs", emb_full_results)

    print("=" * 70)
    print("Final Comparison: All Full Models")
    print("=" * 70)
    print(f"{'Metric':<18} {'BM25 50K':>12} {'BM25 Full':>12} {'Emb Full':>12}")
    print("-" * 55)

    metrics = [
        ("Precision@10", "precision_at_k"),
        ("Recall",       "recall"),
        ("MAP",          "map"),
        ("nDCG@10",      "ndcg_at_k"),
    ]

    for label, key in metrics:
        b50  = bm25_50k_results[key]
        bf   = bm25_full_results[key]
        ef   = emb_full_results[key]
        best = max(b50, bf, ef)

        b50_str = f"{b50:.4f}{'*' if b50 == best else ' '}"
        bf_str  = f"{bf:.4f}{'*' if bf  == best else ' '}"
        ef_str  = f"{ef:.4f}{'*' if ef  == best else ' '}"

        print(f"{label:<18} {b50_str:>12} {bf_str:>12} {ef_str:>12}")

    print()
    print("* = Best in category")


if __name__ == "__main__":
    main()