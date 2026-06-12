import pickle
from itertools import islice

import ir_datasets

from shared.config import DATASET_ID, SAMPLE_SIZE, BM25_SAMPLE_DIR
from services.preprocessing_service.document_preprocessing import preprocess_document
from services.indexing_service.bm25_index import build_bm25_index


BM25_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading dataset...")
    dataset = ir_datasets.load(DATASET_ID)

    print(f"Reading and preprocessing first {SAMPLE_SIZE} documents...")

    doc_ids = []
    tokenized_documents = []

    for doc in islice(dataset.docs_iter(), SAMPLE_SIZE):
        doc_ids.append(doc.doc_id)
        tokenized_documents.append(preprocess_document(doc.text))

        if len(doc_ids) % 1000 == 0:
            print(f"Processed {len(doc_ids)} documents...")

    print("Building BM25 index...")
    bm25 = build_bm25_index(tokenized_documents)

    print("Saving BM25 artifacts...")

    with open(BM25_SAMPLE_DIR / "bm25_index.pkl", "wb") as file:
        pickle.dump(bm25, file)

    with open(BM25_SAMPLE_DIR / "bm25_doc_ids.pkl", "wb") as file:
        pickle.dump(doc_ids, file)

    print("Done.")
    print("Documents indexed:", len(doc_ids))
    print("Artifacts saved in:", BM25_SAMPLE_DIR)


if __name__ == "__main__":
    main()