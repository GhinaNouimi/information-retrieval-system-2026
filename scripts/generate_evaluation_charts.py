import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from shared.config import ARTIFACTS_DIR


CHARTS_DIR = ARTIFACTS_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# النتائج التي حصلنا عليها من التقييم
RESULTS = {
    "TF-IDF\n(10K)": {
        "precision_at_k": 0.0049,
        "recall":         0.0199,
        "map":            0.0181,
        "ndcg_at_k":      0.0214,
    },
    "BM25\n(50K)": {
        "precision_at_k": 0.0203,
        "recall":         0.0977,
        "map":            0.0871,
        "ndcg_at_k":      0.0982,
    },
    "Embedding\n(50K)": {
        "precision_at_k": 0.0230,
        "recall":         0.1033,
        "map":            0.0971,
        "ndcg_at_k":      0.1083,
    },
    "Hybrid\nSerial": {
        "precision_at_k": 0.0225,
        "recall":         0.1023,
        "map":            0.0965,
        "ndcg_at_k":      0.1075,
    },
    "Hybrid\nParallel": {
        "precision_at_k": 0.0220,
        "recall":         0.1018,
        "map":            0.0930,
        "ndcg_at_k":      0.1045,
    },
    "Embedding\nFull(522K)": {
        "precision_at_k": 0.1337,
        "recall":         0.9503,
        "map":            0.8363,
        "ndcg_at_k":      0.8755,
    },
}

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#E377C2"]
MODELS   = list(RESULTS.keys())
METRICS  = {
    "precision_at_k": "Precision@10",
    "recall":         "Recall",
    "map":            "MAP",
    "ndcg_at_k":      "nDCG@10",
}


def chart_1_grouped_bar():
    """
    رسم بياني شريطي مجمّع يقارن كل النماذج في كل المقاييس.
    """
    x      = np.arange(len(METRICS))
    width  = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (model, values) in enumerate(RESULTS.items()):
        metric_values = [values[m] for m in METRICS]
        offset = (i - len(MODELS) / 2) * width + width / 2
        bars = ax.bar(x + offset, metric_values, width, label=model, color=COLORS[i])
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7
            )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — All Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS.values(), fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(v for r in RESULTS.values() for v in r.values()) * 1.3)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = CHARTS_DIR / "chart_1_grouped_bar.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def chart_2_per_metric():
    """
    أربعة رسومات منفصلة — رسم لكل مقياس.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (metric_key, metric_label) in enumerate(METRICS.items()):
        values = [RESULTS[m][metric_key] for m in MODELS]
        bars   = axes[idx].bar(MODELS, values, color=COLORS)

        for bar, val in zip(bars, values):
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=8
            )

        axes[idx].set_title(metric_label, fontsize=12, fontweight="bold")
        axes[idx].set_ylabel("Score", fontsize=10)
        axes[idx].set_ylim(0, max(values) * 1.3)
        axes[idx].grid(axis="y", alpha=0.3)
        axes[idx].tick_params(axis="x", labelsize=8)

    plt.suptitle("Evaluation Results per Metric", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = CHARTS_DIR / "chart_2_per_metric.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def chart_3_radar():
    """
    Radar Chart يعرض أداء كل نموذج على شكل مضلع.
    """
    metric_labels = list(METRICS.values())
    num_metrics   = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, (model, values) in enumerate(RESULTS.items()):
        vals = [values[m] for m in METRICS]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=model, color=COLORS[i])
        ax.fill(angles, vals, alpha=0.1, color=COLORS[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_title("Model Performance — Radar Chart", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = CHARTS_DIR / "chart_3_radar.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def chart_4_sample_size():
    """
    رسم يوضح تأثير حجم العينة على أداء BM25.
    """
    sizes  = ["10K docs\n(TF-IDF)", "10K docs\n(BM25)", "50K docs\n(BM25)"]
    map_v  = [0.0181, 0.0191, 0.0871]
    ndcg_v = [0.0214, 0.0225, 0.0982]

    x     = np.arange(len(sizes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))

    bars1 = ax.bar(x - width/2, map_v,  width, label="MAP",     color="#4C72B0")
    bars2 = ax.bar(x + width/2, ndcg_v, width, label="nDCG@10", color="#DD8452")

    for bar in list(bars1) + list(bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_xlabel("Model & Sample Size", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Impact of Sample Size on Performance", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(ndcg_v) * 1.3)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = CHARTS_DIR / "chart_4_sample_size.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    print("Generating evaluation charts...")
    print()

    chart_1_grouped_bar()
    chart_2_per_metric()
    chart_3_radar()
    chart_4_sample_size()

    print()
    print("All charts saved in:", CHARTS_DIR)
    print()
    print("Charts generated:")
    print("  chart_1_grouped_bar.png  — مقارنة كل النماذج في كل المقاييس")
    print("  chart_2_per_metric.png   — رسم منفصل لكل مقياس")
    print("  chart_3_radar.png        — Radar Chart لكل النماذج")
    print("  chart_4_sample_size.png  — تأثير حجم العينة على الأداء")


if __name__ == "__main__":
    main()