import json
import pickle

from shared.config import ARTIFACTS_DIR


CLUSTERING_FULL_DIR = ARTIFACTS_DIR / "clustering_full"


class ClusteringService:
    def __init__(self):
        with open(CLUSTERING_FULL_DIR / "doc_cluster_assignments.pkl", "rb") as file:
            self.doc_to_cluster = pickle.load(file)

        with open(CLUSTERING_FULL_DIR / "cluster_labels.json", "r", encoding="utf-8") as file:
            raw_labels = json.load(file)
        # JSON object keys are strings; convert back to int cluster ids.
        self.cluster_labels = {int(k): v for k, v in raw_labels.items()}

    def get_cluster_for_doc(self, doc_id):

        return self.doc_to_cluster.get(doc_id)

    def get_label_for_cluster(self, cluster_id):
        terms = self.cluster_labels.get(cluster_id, {}).get("top_terms", [])
        return ", ".join(terms[:5]) if terms else f"Cluster {cluster_id}"

    def group_results(self, results):
       
        groups = {}

        for result in results:
            cluster_id = self.get_cluster_for_doc(result["doc_id"])
            if cluster_id is None:
                cluster_id = -1  # not covered by the clustering index
            groups.setdefault(cluster_id, []).append(result)

        grouped = []
        for cluster_id, items in groups.items():
            label = "Uncategorized" if cluster_id == -1 else self.get_label_for_cluster(cluster_id)
            grouped.append({"cluster_id": cluster_id, "label": label, "results": items})

        grouped.sort(key=lambda g: min(item.get("rank", 0) for item in g["results"]))
        return grouped
