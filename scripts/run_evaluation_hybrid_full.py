
from services.evaluation_service.evaluator import evaluate
from services.retrieval_service.strategies.hybrid_serial_full_v2_strategy import HybridSerialFullV2RetrievalStrategy
from services.retrieval_service.strategies.hybrid_parallel_full_strategy import HybridParallelFullRetrievalStrategy


def print_results(model_name: str, results: dict):
    print("=" * 55)
    print(f"  Model: {model_name}")
    print("=" * 55)
    print(f"  Queries evaluated : {results['num_queries_evaluated']:,}")
    print(f"  Precision@10      : {results['precision_at_k']:.4f}")
    print(f"  Recall            : {results['recall']:.4f}")
    print(f"  MAP               : {results['map']:.4f}")
    print(f"  nDCG@10           : {results['ndcg_at_k']:.4f}")
    print()


def main():
    print("=" * 55)
    print("   Hybrid Full Evaluation (522,931 documents)")
    print("=" * 55)
    print()

    print("[1/2] Loading Hybrid Serial Full...")
    serial = HybridSerialFullV2RetrievalStrategy()
    print()

    print("[2/2] Loading Hybrid Parallel Full...")
    parallel = HybridParallelFullRetrievalStrategy()
    print()

    print("Both strategies loaded. Starting evaluation...")
    print()

    print("Evaluating Hybrid Serial Full...")
    print("(قد يأخذ وقتاً — BM25 يبحث في 522K ثم Embedding يعيد الترتيب)")
    print()
    serial_results = evaluate(serial, top_k=10)
    print_results("Hybrid Serial Full (522K)", serial_results)

    print("Evaluating Hybrid Parallel Full...")
    print("(قد يأخذ وقتاً — كلا النموذجين يبحثان في 522K)")
    print()
    parallel_results = evaluate(parallel, top_k=10)
    print_results("Hybrid Parallel Full (522K)", parallel_results)
    print()
    print("=" * 75)
    print("   Final Comparison: All Models")
    print("=" * 75)
    print(f"  {'النموذج':<28} {'P@10':>7} {'Recall':>8} {'MAP':>8} {'nDCG@10':>9}")
    print(f"  {'-'*28} {'-'*7} {'-'*8} {'-'*8} {'-'*9}")
    print(f"  {'TF-IDF (10K)':<28} {'0.0049':>7} {'0.0199':>8} {'0.0181':>8} {'0.0214':>9}")
    print(f"  {'BM25 (50K)':<28} {'0.0203':>7} {'0.0977':>8} {'0.0871':>8} {'0.0982':>9}")
    print(f"  {'Embedding (50K)':<28} {'0.0230':>7} {'0.1033':>8} {'0.0971':>8} {'0.1083':>9}")
    print(f"  {'Hybrid Serial (50K)':<28} {'0.0225':>7} {'0.1023':>8} {'0.0965':>8} {'0.1075':>9}")
    print(f"  {'Hybrid Parallel (50K)':<28} {'0.0220':>7} {'0.1018':>8} {'0.0930':>8} {'0.1045':>9}")
    print(f"  {'TF-IDF Full v2 (522K)':<28} {'0.1031':>7} {'0.7747':>8} {'0.6134':>8} {'0.6654':>9}")
    print(f"  {'BM25 Full (522K)':<28} {'0.1162':>7} {'0.8638':>8} {'0.7154':>8} {'0.7646':>9}")
    print(f"  {'Embedding Full (522K)':<28} {'0.1337':>7} {'0.9503':>8} {'0.8363':>8} {'0.8755':>9}")
    print(f"  {'Hybrid Serial Full (522K)':<28} {serial_results['precision_at_k']:>7.4f} {serial_results['recall']:>8.4f} {serial_results['map']:>8.4f} {serial_results['ndcg_at_k']:>9.4f}")
    print(f"  {'Hybrid Parallel Full (522K)':<28} {parallel_results['precision_at_k']:>7.4f} {parallel_results['recall']:>8.4f} {parallel_results['map']:>8.4f} {parallel_results['ndcg_at_k']:>9.4f}")
    print()
    print("  ملاحظة: النماذج على 50K تبحث في عينة من الـ Dataset،")
    print("  لذلك قيمها أقل من النماذج على 522K.")


if __name__ == "__main__":
    main()