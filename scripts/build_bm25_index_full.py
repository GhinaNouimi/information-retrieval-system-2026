import pickle

import ir_datasets

from shared.config import DATASET_ID, ARTIFACTS_DIR
from services.preprocessing_service.document_preprocessing import preprocess_document
from services.indexing_service.bm25_index import build_bm25_index


BM25_FULL_DIR = ARTIFACTS_DIR / "bm25_full"
BM25_FULL_DIR.mkdir(parents=True, exist_ok=True)

FULL_SIZE = 522_931


def main():
    print("Loading dataset...")
    dataset = ir_datasets.load(DATASET_ID)

    print(f"Reading and preprocessing all {FULL_SIZE:,} documents...")
    print("This will take 1-2 hours, please wait...")
    print()

    doc_ids = []
    tokenized_documents = []

    for doc in dataset.docs_iter():
        doc_ids.append(doc.doc_id)
        tokenized_documents.append(preprocess_document(doc.text))

        if len(doc_ids) % 10_000 == 0:
            print(f"Processed {len(doc_ids):,} / {FULL_SIZE:,} documents...")

    print()
    print("Building BM25 index...")
    bm25 = build_bm25_index(tokenized_documents)

    print("Saving BM25 full artifacts...")

    with open(BM25_FULL_DIR / "bm25_index.pkl", "wb") as file:
        pickle.dump(bm25, file)

    with open(BM25_FULL_DIR / "bm25_doc_ids.pkl", "wb") as file:
        pickle.dump(doc_ids, file)

    print()
    print("Done.")
    print(f"Documents indexed  : {len(doc_ids):,}")
    print(f"Artifacts saved in : {BM25_FULL_DIR}")


if __name__ == "__main__":
    main()