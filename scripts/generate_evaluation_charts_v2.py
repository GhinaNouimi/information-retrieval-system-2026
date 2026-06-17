import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared.config import ARTIFACTS_DIR


CHARTS_DIR = ARTIFACTS_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ===================================================================
# جميع النتائج — مرتبة من الأضعف للأقوى
# ===================================================================
RESULTS = {
    "TF-IDF\n(10K)": {
        "precision_at_k": 0.0049,
        "recall":         0.0199,
        "map":            0.0181,
        "ndcg_at_k":      0.0214,
        "group":          "sample",
    },
    "BM25\n(50K)": {
        "precision_at_k": 0.0203,
        "recall":         0.0977,
        "map":            0.0871,
        "ndcg_at_k":      0.0982,
        "group":          "sample",
    },
    "Embedding\n(50K)": {
        "precision_at_k": 0.0230,
        "recall":         0.1033,
        "map":            0.0971,
        "ndcg_at_k":      0.1083,
        "group":          "sample",
    },
    "Hybrid Serial\n(50K)": {
        "precision_at_k": 0.0225,
        "recall":         0.1023,
        "map":            0.0965,
        "ndcg_at_k":      0.1075,
        "group":          "sample",
    },
    "Hybrid Parallel\n(50K)": {
        "precision_at_k": 0.0220,
        "recall":         0.1018,
        "map":            0.0930,
        "ndcg_at_k":      0.1045,
        "group":          "sample",
    },
    "TF-IDF Full\n(522K)": {
        "precision_at_k": 0.1031,
        "recall":         0.7747,
        "map":            0.6134,
        "ndcg_at_k":      0.6654,
        "group":          "full",
    },
    "BM25 Full\n(522K)": {
        "precision_at_k": 0.1162,
        "recall":         0.8638,
        "map":            0.7154,
        "ndcg_at_k":      0.7646,
        "group":          "full",
    },
    "Hybrid Parallel\nFull (522K)": {
        "precision_at_k": 0.1294,
        "recall":         0.9345,
        "map":            0.8005,
        "ndcg_at_k":      0.8458,
        "group":          "full",
    },
    "Hybrid Serial\nFull (522K)": {
        "precision_at_k": 0.1311,
        "recall":         0.9379,
        "map":            0.8326,
        "ndcg_at_k":      0.8701,
        "group":          "full",
    },
    "Embedding Full\n(522K)": {
        "precision_at_k": 0.1337,
        "recall":         0.9503,
        "map":            0.8363,
        "ndcg_at_k":      0.8755,
        "group":          "full",
    },
}

METRICS = {
    "precision_at_k": "Precision@10",
    "recall":         "Recall",
    "map":            "MAP",
    "ndcg_at_k":      "nDCG@10",
}

MODELS = list(RESULTS.keys())

# لون مختلف لكل مجموعة — Sample بألوان فاتحة، Full بألوان داكنة
GROUP_COLORS = {
    "sample": ["#A8C4E0", "#F4B88A", "#90CFA8", "#E89898", "#B8ACD8"],
    "full":   ["#2166AC", "#D6604D", "#1A7A3A", "#8B0000", "#4A90D9"],
}

def _get_colors():
    """يُرجع قائمة ألوان بناءً على مجموعة كل نموذج."""
    sample_iter = iter(GROUP_COLORS["sample"])
    full_iter   = iter(GROUP_COLORS["full"])
    colors = []
    for model in MODELS:
        group = RESULTS[model]["group"]
        colors.append(next(sample_iter) if group == "sample" else next(full_iter))
    return colors

COLORS = _get_colors()


# ===================================================================
# Chart 1: Grouped Bar — كل النماذج في كل المقاييس
# ===================================================================
def chart_1_grouped_bar():
    x     = np.arange(len(METRICS))
    width = 0.08
    fig, ax = plt.subplots(figsize=(16, 7))

    for i, (model, values) in enumerate(RESULTS.items()):
        metric_values = [values[m] for m in METRICS]
        offset = (i - len(MODELS) / 2) * width + width / 2
        bars = ax.bar(x + offset, metric_values, width, label=model, color=COLORS[i])
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.005,
                    f"{h:.2f}",
                    ha="center", va="bottom", fontsize=6, rotation=90
                )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — All Metrics (Sample vs Full Dataset)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS.values(), fontsize=12)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = CHARTS_DIR / "chart_1_grouped_bar.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ===================================================================
# Chart 2: Per Metric — رسم منفصل لكل مقياس
# ===================================================================
def chart_2_per_metric():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (metric_key, metric_label) in enumerate(METRICS.items()):
        values = [RESULTS[m][metric_key] for m in MODELS]
        bars   = axes[idx].bar(MODELS, values, color=COLORS)

        for bar, val in zip(bars, values):
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=7, rotation=45
            )

        axes[idx].set_title(metric_label, fontsize=13, fontweight="bold")
        axes[idx].set_ylabel("Score", fontsize=10)
        axes[idx].set_ylim(0, min(max(values) * 1.35, 1.1))
        axes[idx].grid(axis="y", alpha=0.3)
        axes[idx].tick_params(axis="x", labelsize=7)

        # خط فاصل بين Sample و Full
        axes[idx].axvline(x=4.5, color="red", linestyle="--", alpha=0.4, linewidth=1.5)
        axes[idx].text(2.0, max(values) * 1.18, "Sample (50K)", fontsize=8,
                       color="gray", ha="center")
        axes[idx].text(7.0, max(values) * 1.18, "Full (522K)", fontsize=8,
                       color="gray", ha="center")

    plt.suptitle("Evaluation Results per Metric", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = CHARTS_DIR / "chart_2_per_metric.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ===================================================================
# Chart 3: Radar — نماذج Full فقط (لأن Sample صغيرة جداً تُشوّه الشكل)
# ===================================================================
def chart_3_radar():
    full_models = {k: v for k, v in RESULTS.items() if v["group"] == "full"}
    full_colors = GROUP_COLORS["full"]

    metric_labels = list(METRICS.values())
    num_metrics   = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for i, (model, values) in enumerate(full_models.items()):
        vals = [values[m] for m in METRICS]
        vals += vals[:1]
        label = model.replace("\n", " ")
        ax.plot(angles, vals, "o-", linewidth=2, label=label, color=full_colors[i])
        ax.fill(angles, vals, alpha=0.08, color=full_colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title("Full Dataset Models — Radar Chart", fontsize=14,
                 fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = CHARTS_DIR / "chart_3_radar.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ===================================================================
# Chart 4: Sample Size Impact — تأثير حجم البيانات على nDCG@10
# ===================================================================
def chart_4_sample_size():
    """
    يوضح كيف يتحسن الأداء بشكل كبير عند الانتقال من 50K إلى 522K.
    """
    categories = [
        "TF-IDF\n(10K)",
        "BM25\n(50K)",
        "Embedding\n(50K)",
        "TF-IDF Full\n(522K)",
        "BM25 Full\n(522K)",
        "Embedding Full\n(522K)",
    ]
    map_vals  = [0.0181, 0.0871, 0.0971, 0.6134, 0.7154, 0.8363]
    ndcg_vals = [0.0214, 0.0982, 0.1083, 0.6654, 0.7646, 0.8755]
    bar_colors = [
        "#A8C4E0", "#A8C4E0", "#A8C4E0",   # Sample — فاتح
        "#2166AC", "#2166AC", "#2166AC",   # Full — داكن
    ]

    x     = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width / 2, map_vals,  width, label="MAP",     color=bar_colors, alpha=0.85)
    bars2 = ax.bar(x + width / 2, ndcg_vals, width, label="nDCG@10", color=bar_colors, alpha=0.55)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.008,
            f"{h:.3f}",
            ha="center", va="bottom", fontsize=8
        )

    # خط فاصل بين Sample و Full
    ax.axvline(x=2.5, color="red", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.text(1.0,  0.85, "Sample\n(10K–50K)", fontsize=10, color="gray", ha="center")
    ax.text(4.0,  0.85, "Full\n(522K)",      fontsize=10, color="#2166AC", ha="center",
            fontweight="bold")

    ax.set_xlabel("Model & Dataset Size", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Impact of Dataset Size on Retrieval Performance", fontsize=14,
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = CHARTS_DIR / "chart_4_sample_size.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ===================================================================
# Chart 5: Hybrid vs Base — مقارنة Hybrid مع مكوناته الأساسية
# ===================================================================
def chart_5_hybrid_comparison():
    """
    يوضح أن Hybrid يجمع قوة BM25 و Embedding معاً.
    """
    models = [
        "BM25 Full\n(522K)",
        "Embedding Full\n(522K)",
        "Hybrid Serial\nFull (522K)",
        "Hybrid Parallel\nFull (522K)",
    ]
    values = {
        "MAP":     [0.7154, 0.8363, 0.8326, 0.8005],
        "nDCG@10": [0.7646, 0.8755, 0.8701, 0.8458],
        "Recall":  [0.8638, 0.9503, 0.9379, 0.9345],
    }
    colors_map = {"MAP": "#4C72B0", "nDCG@10": "#DD8452", "Recall": "#55A868"}

    x     = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, vals) in enumerate(values.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=metric,
                      color=colors_map[metric], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.004,
                f"{h:.3f}",
                ha="center", va="bottom", fontsize=8
            )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Hybrid vs Base Models — Full Dataset (522K)", fontsize=14,
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0.6, 1.02)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = CHARTS_DIR / "chart_5_hybrid_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ===================================================================
# Main
# ===================================================================
def main():
    print("Generating evaluation charts...")
    print()

    chart_1_grouped_bar()
    chart_2_per_metric()
    chart_3_radar()
    chart_4_sample_size()
    chart_5_hybrid_comparison()

    print()
    print(f"All charts saved in: {CHARTS_DIR}")
    print()
    print("Charts generated:")
    print("  chart_1_grouped_bar.png      — مقارنة كل النماذج في كل المقاييس")
    print("  chart_2_per_metric.png       — رسم منفصل لكل مقياس")
    print("  chart_3_radar.png            — Radar Chart لنماذج Full فقط")
    print("  chart_4_sample_size.png      — تأثير حجم البيانات على الأداء")
    print("  chart_5_hybrid_comparison.png — مقارنة Hybrid مع النماذج الأساسية")


if __name__ == "__main__":
    main()