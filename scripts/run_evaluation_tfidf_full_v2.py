"""
تقييم نموذج TF-IDF Full v2 (Sparse Matrix بدون FAISS)
على كامل الـ 522,931 وثيقة
"""
from services.evaluation_service.evaluator import evaluate
from services.retrieval_service.strategies.tfidf_full_v2_strategy import TfidfFullV2RetrievalStrategy


def main():
    print("=" * 55)
    print("   TF-IDF Full v2 Evaluation (522,931 documents)")
    print("=" * 55)
    print()

    # تحميل النموذج
    print("Loading strategy...")
    strategy = TfidfFullV2RetrievalStrategy()
    print()

    # تشغيل التقييم
    print("Starting evaluation on 10,000 queries...")
    print("(هذا قد يأخذ بعض الوقت — كل استعلام يبحث في 522K وثيقة)")
    print()

    results = evaluate(strategy, top_k=10)

    # عرض النتائج
    print()
    print("=" * 55)
    print("   Results: TF-IDF Full v2")
    print("=" * 55)
    print(f"  Queries evaluated : {results['num_queries_evaluated']:,}")
    print(f"  Precision@10      : {results['precision_at_k']:.4f}")
    print(f"  Recall            : {results['recall']:.4f}")
    print(f"  MAP               : {results['map']:.4f}")
    print(f"  nDCG@10           : {results['ndcg_at_k']:.4f}")
    print("=" * 55)

    # مقارنة سريعة مع النماذج الأخرى للسياق
    print()
    print("مقارنة مع النماذج الأخرى:")
    print(f"  {'النموذج':<25} {'P@10':>6} {'Recall':>8} {'MAP':>8} {'nDCG@10':>9}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*9}")
    print(f"  {'TF-IDF (10K)':<25} {'0.0049':>6} {'0.0199':>8} {'0.0181':>8} {'0.0214':>9}")
    print(f"  {'BM25 (50K)':<25} {'0.0203':>6} {'0.0977':>8} {'0.0871':>8} {'0.0982':>9}")
    print(f"  {'Embedding (50K)':<25} {'0.0230':>6} {'0.1033':>8} {'0.0971':>8} {'0.1083':>9}")
    print(f"  {'BM25 Full (522K)':<25} {'0.1162':>6} {'0.8638':>8} {'0.7154':>8} {'0.7646':>9}")
    print(f"  {'Embedding Full (522K)':<25} {'0.1337':>6} {'0.9503':>8} {'0.8363':>8} {'0.8755':>9}")
    print(f"  {'TF-IDF Full v2 (522K)':<25} {results['precision_at_k']:>6.4f} {results['recall']:>8.4f} {results['map']:>8.4f} {results['ndcg_at_k']:>9.4f}")
    print()


if __name__ == "__main__":
    main()