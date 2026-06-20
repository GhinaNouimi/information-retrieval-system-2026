import gradio as gr

from services.retrieval_service.strategies.bm25_large_strategy import BM25LargeRetrievalStrategy
from services.retrieval_service.strategies.embedding_strategy import EmbeddingRetrievalStrategy
from services.retrieval_service.strategies.embedding_full_strategy import EmbeddingFullRetrievalStrategy
from services.retrieval_service.strategies.hybrid_serial_strategy import HybridSerialRetrievalStrategy
from services.retrieval_service.strategies.hybrid_parallel_strategy import HybridParallelRetrievalStrategy
from services.retrieval_service.strategies.bm25_full_strategy import BM25FullRetrievalStrategy
from services.retrieval_service.strategies.tfidf_full_v2_strategy import TfidfFullV2RetrievalStrategy
from services.retrieval_service.strategies.hybrid_parallel_full_strategy import HybridParallelFullRetrievalStrategy
from services.retrieval_service.strategies.hybrid_serial_full_v2_strategy import HybridSerialFullV2RetrievalStrategy
from services.query_refinement_service.query_expander import refine_query
from services.clustering_service.clustering_service import ClusteringService


print("Loading strategies... please wait.")
STRATEGIES = {
    "BM25 (50K docs)":                 BM25LargeRetrievalStrategy(),
    "Hybrid Serial (50K docs)":         HybridSerialRetrievalStrategy(),
    "Hybrid Parallel (50K docs)":       HybridParallelRetrievalStrategy(),
    "Embedding (50K docs)":             EmbeddingRetrievalStrategy(),
    "TF-IDF Full v2 (522K docs)":       TfidfFullV2RetrievalStrategy(),
    "BM25 Full (522K docs)":            BM25FullRetrievalStrategy(),
    "Embedding Full (522K docs)":       EmbeddingFullRetrievalStrategy(),
    "Hybrid Serial Full (522K docs)":   HybridSerialFullV2RetrievalStrategy(),
    "Hybrid Parallel Full (522K docs)": HybridParallelFullRetrievalStrategy(),
}
print("All strategies loaded.")


print("Loading clustering service...")
clustering_service = ClusteringService()
print("Clustering service loaded.")

# النماذج التي تستخدم BM25
BM25_MODELS = {
    "BM25 (50K docs)",
    "BM25 Full (522K docs)",
    "Hybrid Serial (50K docs)",
    "Hybrid Serial Full (522K docs)",
    "Hybrid Parallel (50K docs)",
    "Hybrid Parallel Full (522K docs)",
}


def search(query: str, model_name: str, use_refinement: bool, use_clustering: bool):
    if not query.strip():
        return "Please enter a query.", ""

    refined_info = ""
    final_query  = query

    if use_refinement:
        refined     = refine_query(query)
        final_query = refined["final"]
        if refined["corrected"] != refined["original"]:
            refined_info += f"**Spell Correction:** {refined['original']} → {refined['corrected']}\n\n"
        refined_info += f"**Expanded Query:** {refined['expanded']}"

    strategy = STRATEGIES[model_name]
    results  = strategy.search(final_query)

    output = ""

    if use_clustering:
        groups = clustering_service.group_results(results)
        for group in groups:
            output += f"## 🗂️ {group['label']} ({len(group['results'])} results)\n\n"
            for result in group["results"]:
                output += f"**Rank {result['rank']}** — Score: {result['score']:.4f}\n"
                output += f"Doc ID: {result['doc_id']}\n"
                output += f"{result['text']}\n"
                output += "---\n"
            output += "\n"
    else:
        for result in results:
            output += f"**Rank {result['rank']}** — Score: {result['score']:.4f}\n"
            output += f"Doc ID: {result['doc_id']}\n"
            output += f"{result['text']}\n"
            output += "---\n"

    return output, refined_info


def explain_bm25(query: str, model_name: str):
    """
    يشرح تأثير المعاملات على الاستعلام الحالي.
    يعمل فقط عند اختيار نموذج BM25.
    """
    if model_name not in BM25_MODELS:
        return "ℹ️ BM25 parameter explanation is only available for BM25 models."

    if not query.strip():
        return "Please enter a query first."

    explanation = f"""
## BM25 Parameters — Current Query Analysis

**Query:** `{query}`
**Model:** {model_name}

---

### المعاملات المستخدمة

| المعامل | القيمة | التأثير |
|---|---|---|
| **k1** | 1.5 | تحكم في تأثير تكرار الكلمة |
| **b**  | 0.75 | تحكم في تأثير طول الوثيقة |

---

### كيف يحسب BM25 درجة الوثيقة؟

$$score(D, Q) = \\sum_{{t \\in Q}} IDF(t) \\cdot \\frac{{tf(t,D) \\cdot (k1 + 1)}}{{tf(t,D) + k1 \\cdot (1 - b + b \\cdot |D|/avgdl)}}$$

---

### تأثير k1 = 1.5

| k1 | التأثير |
|---|---|
| 0.5 | تكرار الكلمة لا يؤثر كثيراً — مناسب لوثائق قصيرة |
| **1.5** ✓ | توازن مثالي — القيمة الموصى بها في الأبحاث |
| 2.5 | تكرار الكلمة يرفع الدرجة أكثر — مناسب لوثائق طويلة |

### تأثير b = 0.75

| b | التأثير |
|---|---|
| 0.0 | طول الوثيقة لا يؤثر على الدرجة |
| **0.75** ✓ | تطبيع جزئي — القيمة الموصى بها في الأبحاث |
| 1.0 | تطبيع كامل حسب طول الوثيقة |

---

### لماذا k1=1.5 و b=0.75 لـ Quora Dataset؟

- **Quora** يحتوي أسئلة قصيرة ومتشابهة في الطول → b=0.75 مناسب
- الأسئلة لا تكرر الكلمات كثيراً → k1=1.5 يعطي وزناً معقولاً للتكرار
- هذه القيم هي الـ default الموصى بها في أبحاث IR وأثبتت فعاليتها تجريبياً
"""
    return explanation


def build_ui():
    with gr.Blocks(title="IR Search System") as app:

        gr.Markdown("# Information Retrieval Search System")
        gr.Markdown("Search through **522,931 documents** using different retrieval models.")

        with gr.Row():
            # ====== عمود الاستعلام ======
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Enter your query",
                    placeholder="e.g. how to invest in stock market",
                    lines=2,
                )
            # ====== عمود الإعدادات ======
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    choices=list(STRATEGIES.keys()),
                    value="Embedding Full (522K docs)",
                    label="Retrieval Model",
                )
                refinement_toggle = gr.Checkbox(
                    label="Enable Query Refinement",
                    value=False,
                )

                clustering_toggle = gr.Checkbox(
                    label="Group results by cluster (K-Means)",
                    value=False,
                )

                search_button = gr.Button("Search", variant="primary")

        # ====== النتائج ======
        refinement_output = gr.Markdown(label="Query Refinement Info")
        results_output    = gr.Markdown(label="Search Results")

        # ====== قسم BM25 Parameters ======
        with gr.Accordion("BM25 Parameter Analysis", open=False):
            gr.Markdown("Select a BM25 model and enter a query, then click the button below.")
            explain_button  = gr.Button("Explain BM25 Parameters for This Query")
            bm25_explanation = gr.Markdown()

        # ====== الأحداث ======
        search_button.click(
            fn=search,
            inputs=[query_input, model_selector, refinement_toggle, clustering_toggle],
            outputs=[results_output, refinement_output],
        )

        explain_button.click(
            fn=explain_bm25,
            inputs=[query_input, model_selector],
            outputs=[bm25_explanation],
        )

        gr.Markdown("---")
        gr.Markdown(
            "**Dataset:** beir/quora/test | "
            "**Full Index:** 522,931 docs | "
            "**Sample Index:** 50,000 docs"
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch()