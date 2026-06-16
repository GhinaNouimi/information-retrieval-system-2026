from itertools import islice

import ir_datasets

from shared.config import DATASET_ID
from services.document_store_service.document_database import (
    create_documents_table,
    insert_document,
    get_document_by_id,
)

FULL_SIZE = 522_931


def main():
    print("Loading dataset...")
    dataset = ir_datasets.load(DATASET_ID)

    print("Creating documents table...")
    create_documents_table()

    print(f"Inserting all {FULL_SIZE:,} documents into SQLite database...")
    print("This will take 15-30 minutes, please wait...")
    print()

    inserted_count = 0

    for doc in dataset.docs_iter():
        insert_document(doc.doc_id, doc.text)
        inserted_count += 1

        if inserted_count % 10_000 == 0:
            print(f"Inserted {inserted_count:,} / {FULL_SIZE:,} documents...")

    print()
    print("Done.")
    print(f"Total inserted documents: {inserted_count:,}")

    print("\nTesting retrieval...")
    sample = get_document_by_id("1")
    print(f"Document ID 1: {sample[:80] if sample else 'Not found'}")


if __name__ == "__main__":
    main()