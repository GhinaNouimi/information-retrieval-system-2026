from services.retrieval_service.strategies.embedding_full_strategy import EmbeddingFullRetrievalStrategy


if __name__ == "__main__":
    query = "how can I invest in stock market"

    strategy = EmbeddingFullRetrievalStrategy()
    results = strategy.search(query)

    print("Query:")
    print(query)

    print("\nEmbedding Full (522,931 docs) Top Results:")
    for result in results:
        print(f"\nRank  : {result['rank']}")
        print(f"Doc ID: {result['doc_id']}")
        print(f"Score : {result['score']:.4f}")
        print(f"Text  : {result['text']}")