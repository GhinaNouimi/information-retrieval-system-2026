import pickle

import numpy as np
import faiss
import ir_datasets
from sentence_transformers import SentenceTransformer

from shared.config import DATASET_ID, ARTIFACTS_DIR


EMBEDDING_FULL_DIR = ARTIFACTS_DIR / "embedding_full"
EMBEDDING_FULL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 512
FULL_SIZE  = 522_931


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")
    print()

    print("Loading dataset...")
    dataset = ir_datasets.load(DATASET_ID)

    print(f"Reading all {FULL_SIZE:,} documents...")
    doc_ids  = []
    doc_texts = []

    for doc in dataset.docs_iter():
        doc_ids.append(doc.doc_id)
        doc_texts.append(doc.text)

        if len(doc_ids) % 50_000 == 0:
            print(f"  Read {len(doc_ids):,} / {FULL_SIZE:,} documents...")

    print(f"Documents loaded: {len(doc_ids):,}")
    print()

    print("Building embeddings...")
    print(f"Batch size: {BATCH_SIZE}")
    print("This will take 1-2 hours, please wait...")
    print()

    embeddings = model.encode(
        doc_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print()
    print(f"Embeddings shape: {embeddings.shape}")
    print()

    print("Building FAISS index...")
    dimension = embeddings.shape[1]

    # نستخدم IndexFlatIP للبحث بـ Cosine Similarity
    # نحوّل المتجهات أولاً لتكون normalized
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"FAISS index built. Total vectors: {index.ntotal:,}")
    print()

    print("Saving artifacts...")

    faiss.write_index(index, str(EMBEDDING_FULL_DIR / "faiss_index.bin"))

    with open(EMBEDDING_FULL_DIR / "embedding_doc_ids.pkl", "wb") as file:
        pickle.dump(doc_ids, file)

    print()
    print("Done.")
    print(f"Documents indexed  : {len(doc_ids):,}")
    print(f"Embedding shape    : {embeddings.shape}")
    print(f"Artifacts saved in : {EMBEDDING_FULL_DIR}")


if __name__ == "__main__":
    main()