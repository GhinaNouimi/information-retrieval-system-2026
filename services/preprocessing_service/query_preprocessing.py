from services.preprocessing_service.text_preprocessing import preprocess_text


def preprocess_query(text: str) -> list[str]:
    """
    Preprocesses a user query before retrieval.

    The query must be processed using the same techniques used for
    documents to guarantee compatibility during matching and ranking.
    """

    return preprocess_text(text)


if __name__ == "__main__":
    sample_query = "How can I invest in stock market?"

    print("Original query:")
    print(sample_query)

    print("\nProcessed query:")
    print(preprocess_query(sample_query))