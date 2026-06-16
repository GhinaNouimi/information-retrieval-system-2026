from services.query_refinement_service.query_expander import refine_query
from services.retrieval_service.strategies.bm25_large_strategy import BM25LargeRetrievalStrategy


def demo_refinement_only():
    """
    يعرض تأثير Query Refinement على الاستعلام فقط بدون بحث.
    """
    queries = [
        "investng in stokc market",
        "how to lose wieght fast",
        "best way to lern programming",
        "invest money",
    ]

    print("=" * 55)
    print("Query Refinement Demo")
    print("=" * 55)

    for query in queries:
        result = refine_query(query)
        print()
        print(f"Original  : {result['original']}")
        print(f"Corrected : {result['corrected']}")
        print(f"Expanded  : {result['expanded']}")
        print("-" * 55)


def demo_refinement_with_search():
    """
    يقارن نتائج البحث قبل وبعد Query Refinement.
    """
    query = "investng in stokc market"

    print()
    print("=" * 55)
    print("Search: Before vs After Refinement")
    print("=" * 55)

    refined = refine_query(query)
    print(f"Original Query : {refined['original']}")
    print(f"Refined Query  : {refined['final']}")
    print()

    strategy = BM25LargeRetrievalStrategy()

    print("--- Results BEFORE Refinement ---")
    results_before = strategy.search(refined["original"])
    for result in results_before[:3]:
        print(f"Rank {result['rank']}: {result['text']}")

    print()
    print("--- Results AFTER Refinement ---")
    results_after = strategy.search(refined["final"])
    for result in results_after[:3]:
        print(f"Rank {result['rank']}: {result['text']}")


if __name__ == "__main__":
    demo_refinement_only()
    demo_refinement_with_search()