# import pickle

# import faiss
# import numpy as np
# from sklearn.decomposition import PCA

# from shared.config import ARTIFACTS_DIR


# EMBEDDING_FULL_DIR  = ARTIFACTS_DIR / "embedding_full"
# CLUSTERING_FULL_DIR = ARTIFACTS_DIR / "clustering_full"
# CHARTS_DIR           = ARTIFACTS_DIR / "charts"

# SAMPLE_SIZE = 8_000


# def main():
#     print("Loading FAISS index and reconstructing vectors...")
#     index = faiss.read_index(str(EMBEDDING_FULL_DIR / "faiss_index.bin"))
#     vectors = index.reconstruct_n(0, index.ntotal)
#     vectors = np.ascontiguousarray(vectors, dtype=np.float32)

#     with open(EMBEDDING_FULL_DIR / "embedding_doc_ids.pkl", "rb") as file:
#         doc_ids = pickle.load(file)

#     print("Loading cluster assignments and centroids...")
#     with open(CLUSTERING_FULL_DIR / "doc_cluster_assignments.pkl", "rb") as file:
#         doc_to_cluster = pickle.load(file)
#     centroids = np.load(CLUSTERING_FULL_DIR / "centroids.npy")

#     labels = np.array([doc_to_cluster[d] for d in doc_ids])
#     k = centroids.shape[0]

#     print(f"Sampling {SAMPLE_SIZE:,} documents for visualization...")
#     rng = np.random.default_rng(42)
#     sample_idx = rng.choice(vectors.shape[0], size=min(SAMPLE_SIZE, vectors.shape[0]), replace=False)
#     sample_vectors = vectors[sample_idx]
#     sample_labels  = labels[sample_idx]

#     print("Fitting 2D PCA on the sample...")
#     pca = PCA(n_components=2, random_state=42)
#     sample_2d    = pca.fit_transform(sample_vectors)
#     centroids_2d = pca.transform(centroids)
#     explained = pca.explained_variance_ratio_.sum()
#     print(f"PCA explained variance (2D): {explained:.2%}")

#     print("Plotting...")
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt

#     fig, ax = plt.subplots(figsize=(10, 8))
#     ax.scatter(sample_2d[:, 0], sample_2d[:, 1],
#                c=sample_labels, cmap="nipy_spectral", s=6, alpha=0.5)
#     ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
#                c="black", marker="X", s=120, edgecolors="white", linewidths=1.2,
#                label="Cluster centers")
#     ax.set_title(f"K-Means Clusters (K={k}) — PCA Projection of Embedding Full\n"
#                  f"{SAMPLE_SIZE:,}-doc sample, {explained:.1%} variance explained in 2D")
#     ax.set_xlabel("PCA Component 1")
#     ax.set_ylabel("PCA Component 2")
#     ax.legend(loc="upper right")
#     fig.tight_layout()

#     out_path = CHARTS_DIR / "clustering_pca_scatter.png"
#     plt.savefig(out_path, dpi=150)
#     plt.close(fig)
#     print(f"Saved to {out_path}")


# if __name__ == "__main__":
#     main()


"""
Visualize K-Means clusters from the Embedding Full clustering index using a
2D t-SNE projection. Produces a scatter plot of a sample of documents
colored by cluster, with cluster centroids marked.

Run:
    python -m scripts.plot_clusters_full
"""

import pickle

import faiss
import numpy as np
from sklearn.manifold import TSNE

from shared.config import ARTIFACTS_DIR


EMBEDDING_FULL_DIR  = ARTIFACTS_DIR / "embedding_full"
CLUSTERING_FULL_DIR = ARTIFACTS_DIR / "clustering_full"
CHARTS_DIR           = ARTIFACTS_DIR / "charts"

SAMPLE_SIZE = 5_000  # t-SNE is slower than PCA, so a smaller sample keeps this fast


def main():
    print("Loading FAISS index and reconstructing vectors...")
    index = faiss.read_index(str(EMBEDDING_FULL_DIR / "faiss_index.bin"))
    vectors = index.reconstruct_n(0, index.ntotal)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    with open(EMBEDDING_FULL_DIR / "embedding_doc_ids.pkl", "rb") as file:
        doc_ids = pickle.load(file)

    print("Loading cluster assignments and centroids...")
    with open(CLUSTERING_FULL_DIR / "doc_cluster_assignments.pkl", "rb") as file:
        doc_to_cluster = pickle.load(file)
    centroids = np.load(CLUSTERING_FULL_DIR / "centroids.npy")

    labels = np.array([doc_to_cluster[d] for d in doc_ids])
    k = centroids.shape[0]

    print(f"Sampling {SAMPLE_SIZE:,} documents for visualization...")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(vectors.shape[0], size=min(SAMPLE_SIZE, vectors.shape[0]), replace=False)
    sample_vectors = vectors[sample_idx]
    sample_labels  = labels[sample_idx]

    # t-SNE has no out-of-sample transform, so embed the sample points and
    # the centroids together in one fit, then split the result afterward.
    combined = np.vstack([sample_vectors, centroids])

    print("Fitting 2D t-SNE (this can take a minute or two)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    combined_2d = tsne.fit_transform(combined)

    sample_2d    = combined_2d[:len(sample_vectors)]
    centroids_2d = combined_2d[len(sample_vectors):]

    print("Plotting...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(sample_2d[:, 0], sample_2d[:, 1],
               c=sample_labels, cmap="nipy_spectral", s=8, alpha=0.6)
    ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
               c="black", marker="X", s=140, edgecolors="white", linewidths=1.2,
               label="Cluster centers")
    ax.set_title(f"K-Means Clusters (K={k}) — t-SNE Projection of Embedding Full\n"
                 f"{SAMPLE_SIZE:,}-doc sample")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_path = CHARTS_DIR / "clustering_tsne_scatter.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()