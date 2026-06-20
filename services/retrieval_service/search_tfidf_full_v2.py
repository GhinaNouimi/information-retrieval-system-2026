from services.retrieval_service.strategies.tfidf_full_v2_strategy import TfidfFullV2RetrievalStrategy

if __name__ == "__main__":
    query = "how to invest in stock market"

    strategy = TfidfFullV2RetrievalStrategy()
    results = strategy.search(query)

    print("Query:", query)
    print("\nTF-IDF Full v2 Results:")
    for result in results:
        print(f"\nRank  : {result['rank']}")
        print(f"Doc ID: {result['doc_id']}")
        print(f"Score : {result['score']:.4f}")
        print(f"Text  : {result['text']}")