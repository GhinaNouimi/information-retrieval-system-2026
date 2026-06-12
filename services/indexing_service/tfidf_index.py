from sklearn.feature_extraction.text import TfidfVectorizer

from services.preprocessing_service.document_preprocessing import (
    preprocess_document
)


def build_tfidf_index(documents: dict[str, str]):
    doc_ids = list(documents.keys())

    processed_documents = []

    for text in documents.values():
        tokens = preprocess_document(text)
        processed_documents.append(" ".join(tokens))

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(processed_documents)

    return vectorizer, tfidf_matrix, doc_ids


if __name__ == "__main__":
    sample_documents = {
        "Doc1": "best way invest stock",
        "Doc2": "banana orange fruit",
        "Doc3": "stock market investment guide",
    }

    vectorizer, tfidf_matrix, doc_ids = build_tfidf_index(sample_documents)

    print("Document IDs:")
    print(doc_ids)

    print("\nVocabulary:")
    print(vectorizer.vocabulary_)

    print("\nTF-IDF Matrix Shape:")
    print(tfidf_matrix.shape)