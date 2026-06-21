
import pickle
import ir_datasets

from shared.config import DATASET_ID, ARTIFACTS_DIR
from services.preprocessing_service.document_preprocessing import preprocess_document


BM25_FULL_DIR = ARTIFACTS_DIR / "bm25_full"
BM25_FULL_DIR.mkdir(parents=True, exist_ok=True)

FULL_SIZE = 522_931


def main():
    print("=" * 55)
    print("   Building BM25 Corpus Tokens (522,931 documents)")
    print("=" * 55)
    print()
    print("هذا السكريبت يحفظ الـ tokens فقط بدون بناء index جديد.")
    print()

    print("Loading dataset...")
    dataset = ir_datasets.load(DATASET_ID)

    print(f"Reading and preprocessing all {FULL_SIZE:,} documents...")
    print()

    doc_ids           = []
    corpus_tokens     = []

    for doc in dataset.docs_iter():
        doc_ids.append(doc.doc_id)
        corpus_tokens.append(preprocess_document(doc.text))

        if len(doc_ids) % 10_000 == 0:
            print(f"  Processed {len(doc_ids):,} / {FULL_SIZE:,} documents...")

    print()
    print(f"All {len(doc_ids):,} documents processed.")
    print()

    print("Saving corpus tokens...")
    tokens_path = BM25_FULL_DIR / "bm25_corpus_tokens.pkl"
    with open(tokens_path, "wb") as file:
        pickle.dump(corpus_tokens, file)
    print(f"  Saved: {tokens_path}")

    print()
    print("=" * 55)
    print("Done.")
    print(f"Documents processed : {len(doc_ids):,}")
    print(f"Artifacts saved in  : {BM25_FULL_DIR}")
    print("=" * 55)
    print()


if __name__ == "__main__":
    main()