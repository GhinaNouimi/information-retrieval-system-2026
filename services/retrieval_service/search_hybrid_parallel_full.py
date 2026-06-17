from services.retrieval_service.strategies.hybrid_parallel_full_strategy import HybridParallelFullRetrievalStrategy

if __name__ == "__main__":
    query = "how to invest in stock market"

    strategy = HybridParallelFullRetrievalStrategy()
    results  = strategy.search(query)

    print("Query:", query)
    print("\nHybrid Parallel Full Results:")
    for result in results:
        print(f"\nRank  : {result['rank']}")
        print(f"Doc ID: {result['doc_id']}")
        print(f"Score : {result['score']:.4f}")
        print(f"Text  : {result['text']}")