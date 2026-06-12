from services.retrieval_service.strategies.bm25_strategy import BM25RetrievalStrategy


if __name__ == "__main__":
    query = "how can I invest in stock market"

    strategy = BM25RetrievalStrategy()
    results = strategy.search(query)

    print("Query:")
    print(query)

    print("\nBM25 Top Results:")
    for result in results:
        print("\nRank:", result["rank"])
        print("Doc ID:", result["doc_id"])
        print("Score:", result["score"])
        print("Text:", result["text"])