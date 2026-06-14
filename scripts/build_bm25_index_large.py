import pickle
from itertools import islice

import ir_datasets

from shared.config import DATASET_ID, LARGE_SIZE, BM25_LARGE_DIR
from services.preprocessing_service.document_preprocessing import preprocess_document
from services.indexing_service.bm25_index import build_bm25_index


BM25_LARGE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading dataset...")
    dataset = ir_datasets.load(DATASET_ID)

    print(f"Reading and preprocessing first {LARGE_SIZE} documents...")
    print("This will take several minutes, please wait...")
    print()

    doc_ids = []
    tokenized_documents = []

    for doc in islice(dataset.docs_iter(), LARGE_SIZE):
        doc_ids.append(doc.doc_id)
        tokenized_documents.append(preprocess_document(doc.text))

        if len(doc_ids) % 5000 == 0:
            print(f"Processed {len(doc_ids)} / {LARGE_SIZE} documents...")

    print()
    print("Building BM25 index...")
    bm25 = build_bm25_index(tokenized_documents)

    print("Saving BM25 large artifacts...")

    with open(BM25_LARGE_DIR / "bm25_index.pkl", "wb") as file:
        pickle.dump(bm25, file)

    with open(BM25_LARGE_DIR / "bm25_doc_ids.pkl", "wb") as file:
        pickle.dump(doc_ids, file)

    print()
    print("Done.")
    print(f"Documents indexed : {len(doc_ids)}")
    print(f"Artifacts saved in: {BM25_LARGE_DIR}")


if __name__ == "__main__":
    main()