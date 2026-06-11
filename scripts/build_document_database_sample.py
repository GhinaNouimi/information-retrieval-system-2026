import ir_datasets
from itertools import islice
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from services.document_store_service.document_database import (
    create_documents_table,
    insert_document,
    get_document_by_id,
)


DATASET_ID = "beir/quora/test"
SAMPLE_SIZE = 10_000


def main():
    print("Loading dataset...")
    dataset = ir_datasets.load(DATASET_ID)

    print("Creating documents table...")
    create_documents_table()

    print(f"Inserting first {SAMPLE_SIZE} raw documents into SQLite database...")

    inserted_count = 0

    for doc in islice(dataset.docs_iter(), SAMPLE_SIZE):
        insert_document(doc.doc_id, doc.text)
        inserted_count += 1

        if inserted_count % 1000 == 0:
            print(f"Inserted {inserted_count} documents...")

    print("Done.")
    print("Total inserted documents:", inserted_count)

    print("\nTesting document retrieval from database:")
    sample_doc = get_document_by_id("1")
    print("Document ID: 1")
    print(sample_doc)


if __name__ == "__main__":
    main()