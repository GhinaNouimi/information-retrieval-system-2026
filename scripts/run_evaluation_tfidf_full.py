from services.retrieval_service.strategies.tfidf_strategy import TfidfRetrievalStrategy
from services.retrieval_service.strategies.tfidf_full_strategy import TfidfFullRetrievalStrategy
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
    print("[1/4] Loading TF-IDF Sample (10K)...")
    tfidf_sample = TfidfRetrievalStrategy()

    print("[2/4] Loading TF-IDF Full (522K)...")
    tfidf_full = TfidfFullRetrievalStrategy()

    print("[3/4] Loading BM25 Full (522K)...")
    bm25_full = BM25FullRetrievalStrategy()

    print("[4/4] Loading Embedding Full (522K)...")
    emb_full = EmbeddingFullRetrievalStrategy()

    print()
    print("Starting evaluation...")
    print()

    print("Evaluating TF-IDF Sample (10K)...")
    tfidf_sample_results = evaluate(tfidf_sample)
    print_results("TF-IDF - 10,000 docs", tfidf_sample_results)

    print("Evaluating TF-IDF Full (522K)...")
    tfidf_full_results = evaluate(tfidf_full)
    print_results("TF-IDF Full - 522,931 docs", tfidf_full_results)

    print("Evaluating BM25 Full (522K)...")
    bm25_full_results = evaluate(bm25_full)
    print_results("BM25 Full - 522,931 docs", bm25_full_results)

    print("Evaluating Embedding Full (522K)...")
    emb_full_results = evaluate(emb_full)
    print_results("Embedding Full - 522,931 docs", emb_full_results)

    print("=" * 75)
    print("Final Comparison: All Models on Full Dataset")
    print("=" * 75)
    print(f"{'Metric':<18} {'TF-IDF 10K':>12} {'TF-IDF Full':>12} {'BM25 Full':>12} {'Emb Full':>12}")
    print("-" * 75)

    metrics = [
        ("Precision@10", "precision_at_k"),
        ("Recall",       "recall"),
        ("MAP",          "map"),
        ("nDCG@10",      "ndcg_at_k"),
    ]

    for label, key in metrics:
        ts  = tfidf_sample_results[key]
        tf  = tfidf_full_results[key]
        bf  = bm25_full_results[key]
        ef  = emb_full_results[key]
        best = max(ts, tf, bf, ef)

        ts_str = f"{ts:.4f}{'*' if ts == best else ' '}"
        tf_str = f"{tf:.4f}{'*' if tf == best else ' '}"
        bf_str = f"{bf:.4f}{'*' if bf == best else ' '}"
        ef_str = f"{ef:.4f}{'*' if ef == best else ' '}"

        print(f"{label:<18} {ts_str:>12} {tf_str:>12} {bf_str:>12} {ef_str:>12}")

    print()
    print("* = Best in category")


if __name__ == "__main__":
    main()