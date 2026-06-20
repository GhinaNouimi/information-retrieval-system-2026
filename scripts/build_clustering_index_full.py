

import json
import pickle

import faiss
import numpy as np
import scipy.sparse
from sklearn.metrics import silhouette_score

from shared.config import ARTIFACTS_DIR


EMBEDDING_FULL_DIR  = ARTIFACTS_DIR / "embedding_full"
TFIDF_FULL_V2_DIR   = ARTIFACTS_DIR / "tfidf_full_v2"
CLUSTERING_FULL_DIR = ARTIFACTS_DIR / "clustering_full"
CHARTS_DIR           = ARTIFACTS_DIR / "charts"

CLUSTERING_FULL_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Candidate K values for the elbow sweep. Cheap to run since faiss.Kmeans
K_CANDIDATES = [10, 20, 30, 50, 75, 100, 150]

# Final K used for the production clustering

FINAL_K = 50

TOP_TERMS_PER_CLUSTER  = 10
SILHOUETTE_SAMPLE_SIZE = 5_000


def load_embeddings():
    print("Loading FAISS index and reconstructing vectors...")
    index = faiss.read_index(str(EMBEDDING_FULL_DIR / "faiss_index.bin"))
    vectors = index.reconstruct_n(0, index.ntotal)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    with open(EMBEDDING_FULL_DIR / "embedding_doc_ids.pkl", "rb") as file:
        doc_ids = pickle.load(file)

    print(f"Loaded {vectors.shape[0]:,} vectors of dimension {vectors.shape[1]}")
    return vectors, doc_ids


def run_elbow_sweep(vectors):
    print("\nRunning elbow sweep across candidate K values...")

    rng = np.random.default_rng(42)
    sample_size = min(SILHOUETTE_SAMPLE_SIZE, vectors.shape[0])
    sample_idx = rng.choice(vectors.shape[0], size=sample_size, replace=False)
    sample_vectors = vectors[sample_idx]

    inertias = []
    silhouettes = []

    for k in K_CANDIDATES:
        kmeans = faiss.Kmeans(vectors.shape[1], k, niter=20, verbose=False, seed=42,
                               max_points_per_centroid=vectors.shape[0])
        kmeans.train(vectors)

        inertia = float(kmeans.obj[-1]) if len(kmeans.obj) else None
        inertias.append(inertia)

        _, sample_labels = kmeans.index.search(sample_vectors, 1)
        sample_labels = sample_labels.flatten()
        try:
            sil = float(silhouette_score(sample_vectors, sample_labels))
        except ValueError:
            sil = None
        silhouettes.append(sil)

        print(f"  K={k:>4}  inertia={inertia:.2f}  silhouette(sample)={sil}")

    plot_elbow(K_CANDIDATES, inertias, silhouettes)


def plot_elbow(ks, inertias, silhouettes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ks, inertias, marker="o", color="tab:blue", label="Inertia")
    ax1.set_xlabel("K (number of clusters)")
    ax1.set_ylabel("Inertia", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(ks, silhouettes, marker="s", color="tab:orange", label="Silhouette (sample)")
    ax2.set_ylabel("Silhouette score", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title("K-Means Elbow Sweep — Embedding Full (522K docs)")
    fig.tight_layout()
    out_path = CHARTS_DIR / "clustering_elbow.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nElbow chart saved to {out_path}")


def fit_final_kmeans(vectors, k):
    print(f"\nFitting final K-Means with K={k} on all {vectors.shape[0]:,} vectors...")
    kmeans = faiss.Kmeans(vectors.shape[1], k, niter=50, verbose=True, seed=42,
                           max_points_per_centroid=vectors.shape[0])
    kmeans.train(vectors)

    _, labels = kmeans.index.search(vectors, 1)
    labels = labels.flatten()
    centroids = kmeans.centroids.reshape(k, vectors.shape[1])

    return centroids, labels


def label_clusters(doc_ids, labels, k):
    print("\nLabeling clusters using the existing TF-IDF Full v2 vectorizer...")

    with open(TFIDF_FULL_V2_DIR / "tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
    tfidf_matrix = scipy.sparse.load_npz(TFIDF_FULL_V2_DIR / "tfidf_matrix.npz")
    with open(TFIDF_FULL_V2_DIR / "tfidf_doc_ids.pkl", "rb") as file:
        tfidf_doc_ids = pickle.load(file)

    tfidf_row_of = {doc_id: i for i, doc_id in enumerate(tfidf_doc_ids)}
    feature_names = np.array(vectorizer.get_feature_names_out())

    cluster_labels = {}
    for cluster_id in range(k):
        member_doc_ids = [doc_ids[i] for i in np.where(labels == cluster_id)[0]]
        rows = [tfidf_row_of[d] for d in member_doc_ids if d in tfidf_row_of]

        if not rows:
            cluster_labels[cluster_id] = {"top_terms": [], "size": 0}
            continue

        cluster_tfidf_sum = np.asarray(tfidf_matrix[rows].sum(axis=0)).flatten()
        top_indices = np.argsort(cluster_tfidf_sum)[::-1][:TOP_TERMS_PER_CLUSTER]
        top_terms = feature_names[top_indices].tolist()

        cluster_labels[cluster_id] = {"top_terms": top_terms, "size": len(rows)}
        print(f"  Cluster {cluster_id:>3} ({len(rows):>6} docs): {', '.join(top_terms)}")

    return cluster_labels


def main():
    vectors, doc_ids = load_embeddings()

    run_elbow_sweep(vectors)

    centroids, labels = fit_final_kmeans(vectors, FINAL_K)
    cluster_labels = label_clusters(doc_ids, labels, FINAL_K)

    print("\nSaving artifacts...")
    np.save(CLUSTERING_FULL_DIR / "centroids.npy", centroids)

    doc_to_cluster = {doc_id: int(cid) for doc_id, cid in zip(doc_ids, labels)}
    with open(CLUSTERING_FULL_DIR / "doc_cluster_assignments.pkl", "wb") as file:
        pickle.dump(doc_to_cluster, file)

    with open(CLUSTERING_FULL_DIR / "cluster_labels.json", "w", encoding="utf-8") as file:
        json.dump(cluster_labels, file, ensure_ascii=False, indent=2)

    print(f"\nDone. K={FINAL_K} clusters built over {len(doc_ids):,} documents.")
    print(f"Artifacts saved in: {CLUSTERING_FULL_DIR}")


if __name__ == "__main__":
    main()
