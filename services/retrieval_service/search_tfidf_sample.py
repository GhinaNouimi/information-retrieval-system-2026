import pickle
import sys
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity


TOP_K = 10

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "tfidf_sample"

from services.document_store_service.document_database import get_document_by_id


def load_artifacts():
    with open(ARTIFACT_DIR / "tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    with open(ARTIFACT_DIR / "tfidf_matrix.pkl", "rb") as file:
        tfidf_matrix = pickle.load(file)

    with open(ARTIFACT_DIR / "doc_ids.pkl", "rb") as file:
        doc_ids = pickle.load(file)

    return vectorizer, tfidf_matrix, doc_ids


def search(query: str, top_k: int = TOP_K):
    vectorizer, tfidf_matrix, doc_ids = load_artifacts()

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []

    for rank, index in enumerate(ranked_indices, start=1):
        doc_id = doc_ids[index]
        original_text = get_document_by_id(doc_id)

        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": scores[index],
            "text": original_text
        })

    return results


if __name__ == "__main__":
    query = "how can I invest in stock market"

    results = search(query)

    print("Query:")
    print(query)

    print("\nTop Results:")
    for result in results:
        print("\nRank:", result["rank"])
        print("Doc ID:", result["doc_id"])
        print("Score:", result["score"])
        print("Text:", result["text"])