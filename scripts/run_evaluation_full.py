from services.retrieval_service.strategies.bm25_large_strategy import BM25LargeRetrievalStrategy
from services.retrieval_service.strategies.embedding_strategy import EmbeddingRetrievalStrategy
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
    print("=" * 50)
    print("Full Dataset Evaluation")
    print("=" * 50)
    print()

    print("[1/3] Loading BM25 Large (50K)...")
    bm25 = BM25LargeRetrievalStrategy()

    print("[2/3] Loading Embedding Large (50K)...")
    embedding_50k = EmbeddingRetrievalStrategy()

    print("[3/3] Loading Embedding Full (522,931)...")
    embedding_full = EmbeddingFullRetrievalStrategy()

    print()
    print("All strategies loaded. Starting evaluation...")
    print()

    print("Evaluating BM25 (50K)...")
    bm25_results = evaluate(bm25)
    print_results("BM25 - 50,000 docs", bm25_results)

    print("Evaluating Embedding (50K)...")
    emb_50k_results = evaluate(embedding_50k)
    print_results("Embedding - 50,000 docs", emb_50k_results)

    print("Evaluating Embedding Full (522,931)...")
    emb_full_results = evaluate(embedding_full)
    print_results("Embedding Full - 522,931 docs", emb_full_results)

    # جدول المقارنة النهائي
    print("=" * 65)
    print("Comparison: 50K vs Full Dataset")
    print("=" * 65)
    print(f"{'Metric':<18} {'BM25 50K':>12} {'Emb 50K':>12} {'Emb Full':>12} {'Improvement':>12}")
    print("-" * 65)

    metrics = [
        ("Precision@10", "precision_at_k"),
        ("Recall",       "recall"),
        ("MAP",          "map"),
        ("nDCG@10",      "ndcg_at_k"),
    ]

    for label, key in metrics:
        b   = bm25_results[key]
        e50 = emb_50k_results[key]
        ef  = emb_full_results[key]
        improvement = ((ef - e50) / e50 * 100) if e50 > 0 else 0
        print(f"{label:<18} {b:>12.4f} {e50:>12.4f} {ef:>12.4f} {improvement:>+11.1f}%")

    print()
    print("* Improvement = Embedding Full vs Embedding 50K")


if __name__ == "__main__":
    main()