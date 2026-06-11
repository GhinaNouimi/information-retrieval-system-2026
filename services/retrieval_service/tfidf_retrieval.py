from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from services.indexing_service.tfidf_index import build_tfidf_index


def search_with_tfidf(query: str, documents: dict[str, str], top_k: int = 3):
    vectorizer, tfidf_matrix, doc_ids = build_tfidf_index(documents)

    query_vector = vectorizer.transform([query])

    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    ranked_indices = similarity_scores.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:
        results.append({
            "doc_id": doc_ids[index],
            "score": similarity_scores[index],
            "text": documents[doc_ids[index]]
        })

    return results


if __name__ == "__main__":
    sample_documents = {
        "Doc1": "best way invest stock",
        "Doc2": "banana orange fruit",
        "Doc3": "stock market investment guide",
    }

    query = "invest in stock market"

    results = search_with_tfidf(query, sample_documents, top_k=3)

    print("Query:")
    print(query)

    print("\nTF-IDF Search Results:")
    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print("Doc ID:", result["doc_id"])
        print("Score:", result["score"])
        print("Text:", result["text"])