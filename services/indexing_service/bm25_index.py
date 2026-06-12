from rank_bm25 import BM25Okapi


def build_bm25_index(tokenized_documents: list[list[str]]):
    bm25 = BM25Okapi(tokenized_documents)
    return bm25


if __name__ == "__main__":
    sample_documents = [
        ["best", "way", "invest", "stock"],
        ["banana", "orange", "fruit"],
        ["stock", "market", "investment", "guide"],
    ]

    bm25 = build_bm25_index(sample_documents)

    query_tokens = ["invest", "stock", "market"]

    scores = bm25.get_scores(query_tokens)

    print("BM25 Scores:")
    for index, score in enumerate(scores, start=1):
        print(f"Doc{index}: {score}")