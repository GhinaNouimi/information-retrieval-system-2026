from services.retrieval_service.strategies.hybrid_serial_full_v2_strategy import HybridSerialFullV2RetrievalStrategy

if __name__ == "__main__":
    query = "how to invest in stock market"

    strategy = HybridSerialFullV2RetrievalStrategy()
    results  = strategy.search(query)

    print("Query:", query)
    print("\nHybrid Serial Full Results:")
    for result in results:
        print(f"\nRank  : {result['rank']}")
        print(f"Doc ID: {result['doc_id']}")
        print(f"Score : {result['score']:.4f}")
        print(f"Text  : {result['text']}")