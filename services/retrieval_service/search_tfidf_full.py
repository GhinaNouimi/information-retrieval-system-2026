from services.retrieval_service.strategies.tfidf_full_strategy import TfidfFullRetrievalStrategy

if __name__ == "__main__":
    query = "how to invest in stock market"

    strategy = TfidfFullRetrievalStrategy()
    results = strategy.search(query)

    print("Query:", query)
    print("\nTF-IDF Full Results:")
    for result in results:
        print(f"\nRank  : {result['rank']}")
        print(f"Doc ID: {result['doc_id']}")
        print(f"Score : {result['score']:.4f}")
        print(f"Text  : {result['text']}")