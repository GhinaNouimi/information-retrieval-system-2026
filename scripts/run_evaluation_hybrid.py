from services.retrieval_service.strategies.bm25_large_strategy import BM25LargeRetrievalStrategy
from services.retrieval_service.strategies.embedding_strategy import EmbeddingRetrievalStrategy
from services.retrieval_service.strategies.hybrid_serial_strategy import HybridSerialRetrievalStrategy
from services.retrieval_service.strategies.hybrid_parallel_strategy import HybridParallelRetrievalStrategy
from services.evaluation_service.evaluator import evaluate


def print_results(model_name: str, results: dict):
    print("=" * 45)
    print(f"Model: {model_name}")
    print("=" * 45)
    print(f"Queries Evaluated : {results['num_queries_evaluated']}")
    print(f"Precision@10      : {results['precision_at_k']:.4f}")
    print(f"Recall            : {results['recall']:.4f}")
    print(f"MAP               : {results['map']:.4f}")
    print(f"nDCG@10           : {results['ndcg_at_k']:.4f}")
    print()


def main():
    print("=" * 45)
    print("Loading all strategies...")
    print("=" * 45)
    print()

    print("[1/4] Loading BM25 Large...")
    bm25 = BM25LargeRetrievalStrategy()

    print("[2/4] Loading Embedding...")
    embedding = EmbeddingRetrievalStrategy()

    print("[3/4] Loading Hybrid Serial...")
    hybrid_serial = HybridSerialRetrievalStrategy()

    print("[4/4] Loading Hybrid Parallel...")
    hybrid_parallel = HybridParallelRetrievalStrategy()

    print()
    print("All strategies loaded. Starting evaluation...")
    print()

    # تقييم كل نموذج
    print("Evaluating BM25...")
    bm25_results = evaluate(bm25)
    print_results("BM25", bm25_results)

    print("Evaluating Embedding...")
    embedding_results = evaluate(embedding)
    print_results("Embedding", embedding_results)

    print("Evaluating Hybrid Serial...")
    serial_results = evaluate(hybrid_serial)
    print_results("Hybrid Serial", serial_results)

    print("Evaluating Hybrid Parallel...")
    parallel_results = evaluate(hybrid_parallel)
    print_results("Hybrid Parallel", parallel_results)

    # جدول المقارنة النهائي
    print("=" * 70)
    print("Final Comparison: All Models")
    print("=" * 70)
    print(f"{'Metric':<18} {'BM25':>10} {'Embedding':>12} {'Serial':>10} {'Parallel':>10}")
    print("-" * 70)

    metrics = [
        ("Precision@10", "precision_at_k"),
        ("Recall",       "recall"),
        ("MAP",          "map"),
        ("nDCG@10",      "ndcg_at_k"),
    ]

    for label, key in metrics:
        b = bm25_results[key]
        e = embedding_results[key]
        s = serial_results[key]
        p = parallel_results[key]
        best = max(b, e, s, p)

        # نضع * بجانب الأفضل
        b_str = f"{b:.4f}{'*' if b == best else ' '}"
        e_str = f"{e:.4f}{'*' if e == best else ' '}"
        s_str = f"{s:.4f}{'*' if s == best else ' '}"
        p_str = f"{p:.4f}{'*' if p == best else ' '}"

        print(f"{label:<18} {b_str:>10} {e_str:>12} {s_str:>10} {p_str:>10}")

    print()
    print("* = Best in category")


if __name__ == "__main__":
    main()