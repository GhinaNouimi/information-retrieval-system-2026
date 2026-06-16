import pickle
from itertools import islice

import numpy as np
import ir_datasets
from sentence_transformers import SentenceTransformer

from shared.config import DATASET_ID, LARGE_SIZE, ARTIFACTS_DIR


EMBEDDING_LARGE_DIR = ARTIFACTS_DIR / "embedding_large"
EMBEDDING_LARGE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")
    print()

    print("Loading dataset...")
    dataset = ir_datasets.load(DATASET_ID)

    print(f"Reading first {LARGE_SIZE} documents...")
    doc_ids = []
    doc_texts = []

    for doc in islice(dataset.docs_iter(), LARGE_SIZE):
        doc_ids.append(doc.doc_id)
        doc_texts.append(doc.text)

    print(f"Documents loaded: {len(doc_ids)}")
    print()

    print("Building embeddings...")
    print(f"Batch size: {BATCH_SIZE}")
    print("This will take 10-20 minutes, please wait...")
    print()

    embeddings = model.encode(
        doc_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print()
    print("Saving artifacts...")

    np.save(EMBEDDING_LARGE_DIR / "embeddings_matrix.npy", embeddings)

    with open(EMBEDDING_LARGE_DIR / "embedding_doc_ids.pkl", "wb") as file:
        pickle.dump(doc_ids, file)

    print()
    print("Done.")
    print(f"Documents indexed  : {len(doc_ids)}")
    print(f"Embedding shape    : {embeddings.shape}")
    print(f"Artifacts saved in : {EMBEDDING_LARGE_DIR}")


if __name__ == "__main__":
    main()