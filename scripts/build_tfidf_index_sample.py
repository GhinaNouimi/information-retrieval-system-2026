import ir_datasets
import pickle
from pathlib import Path
from itertools import islice
from sklearn.feature_extraction.text import TfidfVectorizer


DATASET_ID = "beir/quora/test"
SAMPLE_SIZE = 10_000

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "tfidf_sample"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading dataset...")
    dataset = ir_datasets.load(DATASET_ID)

    print(f"Reading first {SAMPLE_SIZE} documents...")

    doc_ids = []
    doc_texts = []

    for doc in islice(dataset.docs_iter(), SAMPLE_SIZE):
        doc_ids.append(doc.doc_id)
        doc_texts.append(doc.text)

    print("Building TF-IDF index...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(doc_texts)

    print("Saving artifacts...")

    with open(ARTIFACT_DIR / "tfidf_vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)

    with open(ARTIFACT_DIR / "tfidf_matrix.pkl", "wb") as file:
        pickle.dump(tfidf_matrix, file)

    with open(ARTIFACT_DIR / "doc_ids.pkl", "wb") as file:
        pickle.dump(doc_ids, file)

    print("Done.")
    print("Documents indexed:", len(doc_ids))
    print("Vocabulary size:", len(vectorizer.vocabulary_))
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    print("Artifacts saved in:", ARTIFACT_DIR)


if __name__ == "__main__":
    main()