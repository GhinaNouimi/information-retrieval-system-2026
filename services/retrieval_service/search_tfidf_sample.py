import pickle
import ir_datasets
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


DATASET_ID = "beir/quora/test"
TOP_K = 10

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "tfidf_sample"


def load_artifacts():
    with open(ARTIFACT_DIR / "tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    with open(ARTIFACT_DIR / "tfidf_matrix.pkl", "rb") as file:
        tfidf_matrix = pickle.load(file)

    with open(ARTIFACT_DIR / "doc_ids.pkl", "rb") as file:
        doc_ids = pickle.load(file)

    return vectorizer, tfidf_matrix, doc_ids


def load_document_texts(doc_ids):
    dataset = ir_datasets.load(DATASET_ID)

    needed_doc_ids = set(doc_ids)
    doc_texts = {}

    for doc in dataset.docs_iter():
        if doc.doc_id in needed_doc_ids:
            doc_texts[doc.doc_id] = doc.text

        if len(doc_texts) == len(needed_doc_ids):
            break

    return doc_texts


def search(query: str, top_k: int = TOP_K):
    vectorizer, tfidf_matrix, doc_ids = load_artifacts()

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    ranked_indices = scores.argsort()[::-1][:top_k]

    doc_texts = load_document_texts(doc_ids)

    results = []

    for rank, index in enumerate(ranked_indices, start=1):
        doc_id = doc_ids[index]
        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": scores[index],
            "text": doc_texts.get(doc_id, "")
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