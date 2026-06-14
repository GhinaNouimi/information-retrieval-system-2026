from services.retrieval_service.strategies.embedding_strategy import EmbeddingRetrievalStrategy


if __name__ == "__main__":
    query = "how can I invest in stock market"

    strategy = EmbeddingRetrievalStrategy()
    results = strategy.search(query)

    print("Query:")
    print(query)

    print("\nEmbedding Top Results:")
    for result in results:
        print("\nRank :", result["rank"])
        print("Doc ID:", result["doc_id"])
        print("Score :", result["score"])
        print("Text  :", result["text"])