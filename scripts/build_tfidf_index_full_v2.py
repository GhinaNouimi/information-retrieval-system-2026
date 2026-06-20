import pickle

import numpy as np
import scipy.sparse
import ir_datasets
from sklearn.feature_extraction.text import TfidfVectorizer

from shared.config import DATASET_ID, ARTIFACTS_DIR


TFIDF_FULL_V2_DIR = ARTIFACTS_DIR / "tfidf_full_v2"
TFIDF_FULL_V2_DIR.mkdir(parents=True, exist_ok=True)

FULL_SIZE    = 522_931
MAX_FEATURES = 50_000   # نرفع عدد الـ features لأن Sparse لا يكلّف ذاكرة كبيرة
BATCH_SIZE   = 10_000   # دفعات أكبر لأن الـ transform سريع على Sparse


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

    # ===================================================================
    # الخطوة 1: بناء TF-IDF Vectorizer وتدريبه على كل الوثائق
    # ===================================================================
    print(f"Building TF-IDF vectorizer (max_features={MAX_FEATURES:,})...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=MAX_FEATURES,
        sublinear_tf=True,      # log(tf) بدل tf — يحسّن الأداء
        dtype=np.float32,
    )

    print("Fitting vectorizer on all documents...")
    vectorizer.fit(doc_texts)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print()

    # ===================================================================
    # الخطوة 2: تحويل الوثائق إلى Sparse Matrix دفعة دفعة
    # ===================================================================
    # لماذا دفعات؟ لأن transform() على 522K وثيقة دفعة واحدة ممكن يضغط الذاكرة
    # لكن كل دفعة تبقى Sparse — لا toarray() هنا إطلاقاً
    # ===================================================================
    print("Transforming documents to TF-IDF sparse matrix in batches...")
    total_batches = (len(doc_texts) + BATCH_SIZE - 1) // BATCH_SIZE
    batch_matrices = []

    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(doc_texts))
        batch = doc_texts[start:end]

        # transform يُرجع Sparse Matrix مباشرة — لا نحوّلها
        batch_sparse = vectorizer.transform(batch)
        batch_matrices.append(batch_sparse)

        if (batch_num + 1) % 5 == 0 or batch_num == total_batches - 1:
            print(f"  Batch {batch_num + 1}/{total_batches} — "
                  f"docs {end:,} transformed so far.")

    # دمج كل الدفعات في مصفوفة Sparse واحدة
    print()
    print("Stacking all batches into one sparse matrix...")
    tfidf_matrix = scipy.sparse.vstack(batch_matrices, format="csr")
    print(f"Sparse matrix shape : {tfidf_matrix.shape}")
    print(f"Non-zero elements   : {tfidf_matrix.nnz:,}")
    density = tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])
    print(f"Density             : {density:.4%}  (كم من المصفوفة غير صفر)")
    print()

    # ===================================================================
    # الخطوة 3: حفظ الـ artifacts
    # ===================================================================
    print("Saving artifacts...")

    # حفظ الـ Sparse Matrix بصيغة .npz — هذه الصيغة مصممة للـ Sparse
    scipy.sparse.save_npz(
        str(TFIDF_FULL_V2_DIR / "tfidf_matrix.npz"),
        tfidf_matrix
    )
    print("  tfidf_matrix.npz saved.")

    # حفظ الـ vectorizer
    with open(TFIDF_FULL_V2_DIR / "tfidf_vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)
    print("  tfidf_vectorizer.pkl saved.")

    # حفظ الـ doc_ids
    with open(TFIDF_FULL_V2_DIR / "tfidf_doc_ids.pkl", "wb") as file:
        pickle.dump(doc_ids, file)
    print("  tfidf_doc_ids.pkl saved.")

    print()
    print("=" * 50)
    print("Done.")
    print(f"Documents indexed  : {len(doc_ids):,}")
    print(f"Vocabulary size    : {len(vectorizer.vocabulary_):,}")
    print(f"Matrix shape       : {tfidf_matrix.shape}")
    print(f"Artifacts saved in : {TFIDF_FULL_V2_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()