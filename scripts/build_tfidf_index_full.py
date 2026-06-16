import pickle

import numpy as np
import faiss
import ir_datasets
from sklearn.feature_extraction.text import TfidfVectorizer

from shared.config import DATASET_ID, ARTIFACTS_DIR


TFIDF_FULL_DIR = ARTIFACTS_DIR / "tfidf_full"
TFIDF_FULL_DIR.mkdir(parents=True, exist_ok=True)

FULL_SIZE    = 522_931
MAX_FEATURES = 10_000    # قلّلنا من 50K إلى 30K لتوفير الذاكرة
BATCH_SIZE   = 2_000     # دفعات أصغر
NLIST        = 50       # عدد الـ clusters في IVF


def main():
    print("Loading dataset...")
    dataset = ir_datasets.load(DATASET_ID)

    print(f"Reading all {FULL_SIZE:,} documents...")
    doc_ids   = []
    doc_texts = []

    for doc in dataset.docs_iter():
        doc_ids.append(doc.doc_id)
        doc_texts.append(doc.text)

        if len(doc_ids) % 50_000 == 0:
            print(f"  Read {len(doc_ids):,} / {FULL_SIZE:,} documents...")

    print(f"Documents loaded: {len(doc_ids):,}")
    print()

    # الخطوة 1: بناء TF-IDF Vectorizer
    print(f"Building TF-IDF vectorizer (max_features={MAX_FEATURES:,})...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )

    print("Fitting vectorizer on all documents...")
    vectorizer.fit(doc_texts)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print()

    # الخطوة 2: بناء training sample لـ IVF
    print("Preparing training sample for FAISS IVF...")
    train_size   = min(50_000, len(doc_texts))
    train_texts  = doc_texts[:train_size]
    train_matrix = vectorizer.transform(train_texts).toarray().astype(np.float32)
    faiss.normalize_L2(train_matrix)

    # بناء IVF index
    dimension  = MAX_FEATURES
    quantizer  = faiss.IndexFlatIP(dimension)
    index      = faiss.IndexIVFFlat(quantizer, dimension, NLIST, faiss.METRIC_INNER_PRODUCT)

    print(f"Training FAISS IVF index on {train_size:,} samples...")
    index.train(train_matrix)
    print("Training complete.")
    print()

    # الخطوة 3: إضافة الوثائق دفعة دفعة
    print("Adding documents to FAISS index in batches...")
    total_batches = (len(doc_texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(doc_texts))
        batch = doc_texts[start:end]

        batch_matrix = vectorizer.transform(batch).toarray().astype(np.float32)
        faiss.normalize_L2(batch_matrix)
        index.add(batch_matrix)

        if (batch_num + 1) % 10 == 0 or batch_num == total_batches - 1:
            print(f"  Batch {batch_num + 1}/{total_batches} — "
                  f"docs {end:,} added so far.")

    print()
    print(f"FAISS index built. Total vectors: {index.ntotal:,}")
    print()

    # الخطوة 4: حفظ الـ artifacts
    print("Saving artifacts...")

    index.nprobe = 10
    faiss.write_index(index, str(TFIDF_FULL_DIR / "faiss_index.bin"))

    with open(TFIDF_FULL_DIR / "tfidf_vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)

    with open(TFIDF_FULL_DIR / "tfidf_doc_ids.pkl", "wb") as file:
        pickle.dump(doc_ids, file)

    print()
    print("Done.")
    print(f"Documents indexed  : {len(doc_ids):,}")
    print(f"Vocabulary size    : {len(vectorizer.vocabulary_):,}")
    print(f"FAISS vectors      : {index.ntotal:,}")
    print(f"Artifacts saved in : {TFIDF_FULL_DIR}")


if __name__ == "__main__":
    main()