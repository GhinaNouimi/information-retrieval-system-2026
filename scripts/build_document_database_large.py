from itertools import islice

import ir_datasets

from shared.config import DATASET_ID, LARGE_SIZE
from services.document_store_service.document_database import (
    create_documents_table,
    insert_document,
    get_document_by_id,
)


def main():
    print("Loading dataset...")
    dataset = ir_datasets.load(DATASET_ID)

    print("Creating documents table...")
    create_documents_table()

    print(f"Inserting first {LARGE_SIZE} documents into SQLite database...")
    print("This will take a few minutes...")
    print()

    inserted_count = 0

    for doc in islice(dataset.docs_iter(), LARGE_SIZE):
        insert_document(doc.doc_id, doc.text)
        inserted_count += 1

        if inserted_count % 5000 == 0:
            print(f"Inserted {inserted_count} / {LARGE_SIZE} documents...")

    print()
    print("Done.")
    print(f"Total inserted documents: {inserted_count}")

    print("\nTesting document retrieval from database:")
    sample_doc = get_document_by_id("28373")
    print("Document ID: 28373")
    print(sample_doc[:100] if sample_doc else "Not found")


if __name__ == "__main__":
    main()