
import gradio as gr

from services.retrieval_service.strategies.bm25_large_strategy import BM25LargeRetrievalStrategy
from services.retrieval_service.strategies.embedding_strategy import EmbeddingRetrievalStrategy
from services.retrieval_service.strategies.embedding_full_strategy import EmbeddingFullRetrievalStrategy
from services.retrieval_service.strategies.hybrid_serial_strategy import HybridSerialRetrievalStrategy
from services.retrieval_service.strategies.hybrid_parallel_strategy import HybridParallelRetrievalStrategy
from services.retrieval_service.strategies.bm25_full_strategy import BM25FullRetrievalStrategy
from services.retrieval_service.strategies.bm25_full_tunable_strategy import BM25FullTunableStrategy
from services.retrieval_service.strategies.tfidf_full_v2_strategy import TfidfFullV2RetrievalStrategy
from services.retrieval_service.strategies.hybrid_parallel_full_strategy import HybridParallelFullRetrievalStrategy
from services.retrieval_service.strategies.hybrid_serial_full_v2_strategy import HybridSerialFullV2RetrievalStrategy
from services.query_refinement_service.query_expander import refine_query
from services.preprocessing_service.query_preprocessing import preprocess_query

print("Loading strategies... please wait.")

BM25_TUNABLE = BM25FullTunableStrategy()

STRATEGIES = {
    "BM25 (50K docs)":                  BM25LargeRetrievalStrategy(),
    "Hybrid Serial (50K docs)":          HybridSerialRetrievalStrategy(),
    "Hybrid Parallel (50K docs)":        HybridParallelRetrievalStrategy(),
    "Embedding (50K docs)":              EmbeddingRetrievalStrategy(),
    "TF-IDF Full v2 (522K docs)":        TfidfFullV2RetrievalStrategy(),
    "BM25 Full (522K docs)":             BM25FullRetrievalStrategy(),
    "BM25 Full - Custom k1/b (522K)":    BM25_TUNABLE,
    "Embedding Full (522K docs)":        EmbeddingFullRetrievalStrategy(),
    "Hybrid Serial Full (522K docs)":    HybridSerialFullV2RetrievalStrategy(),
    "Hybrid Parallel Full (522K docs)":  HybridParallelFullRetrievalStrategy(),
}

print("All strategies loaded.")


def search(query: str, model_name: str, use_refinement: bool,
           k1: float, b: float):

    if not query.strip():
        return "Please enter a query.", ""

    # Query Refinement
    refined_info = ""
    final_query  = query

    if use_refinement:
        refined     = refine_query(query)
        final_query = refined["final"]
        if refined["corrected"] != refined["original"]:
            refined_info += f"**Spell Correction:** {refined['original']} → {refined['corrected']}\n\n"
        refined_info += f"**Expanded Query:** {refined['expanded']}"

    # البحث
    if model_name == "BM25 Full - Custom k1/b (522K)":
        # نمرر الـ tokens وليس النص الخام
        query_tokens = preprocess_query(final_query)
        results      = BM25_TUNABLE.search_with_params(
                           query_tokens, k1=k1, b=b)
        refined_info += f"\n\n*BM25 Parameters: k1={k1}, b={b}*"
    else:
        results = STRATEGIES[model_name].search(final_query)

    # تنسيق النتائج
    output = ""
    for result in results:
        output += f"**Rank {result['rank']}** — Score: {result['score']:.4f}\n"
        output += f"Doc ID: {result['doc_id']}\n"
        output += f"{result['text']}\n"
        output += "---\n"

    return output, refined_info.strip()


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
                search_button = gr.Button("Search", variant="primary")

        # ====== قسم BM25 Parameters ======
        with gr.Accordion("⚙️ BM25 Parameters — Custom k1 / b", open=False):
            gr.Markdown(
                "هذه المعاملات تعمل فقط عند اختيار **BM25 Full - Custom k1/b (522K)** من القائمة.\n\n"
                "- **k1**: يتحكم في تأثير تكرار الكلمة — قيمة أعلى = تكرار الكلمة يؤثر أكثر\n"
                "- **b**: يتحكم في تأثير طول الوثيقة — `b=0` يتجاهل الطول، `b=1` يطبّع كاملاً\n\n"
                "القيم الافتراضية الموصى بها في الأبحاث: **k1=1.5** و **b=0.75**"
            )
            with gr.Row():
                k1_slider = gr.Slider(
                    minimum=0.5, maximum=3.0,
                    value=1.5, step=0.1,
                    label="k1  (default: 1.5)",
                )
                b_slider = gr.Slider(
                    minimum=0.0, maximum=1.0,
                    value=0.75, step=0.05,
                    label="b   (default: 0.75)",
                )
            gr.Markdown(
                "| k1 | التأثير |\n"
                "|---|---|\n"
                "| 0.5 | تكرار الكلمة لا يؤثر كثيراً — مناسب لوثائق قصيرة |\n"
                "| **1.5** ✓ | توازن مثالي — الافتراضي الموصى به |\n"
                "| 2.5 | تكرار الكلمة يرفع الدرجة أكثر — مناسب لوثائق طويلة |\n\n"
                "| b | التأثير |\n"
                "|---|---|\n"
                "| 0.0 | طول الوثيقة لا يؤثر |\n"
                "| **0.75** ✓ | تطبيع جزئي — الافتراضي الموصى به |\n"
                "| 1.0 | تطبيع كامل حسب طول الوثيقة |"
            )

        # ====== النتائج ======
        refinement_output = gr.Markdown(label="Query Refinement Info")
        results_output    = gr.Markdown(label="Search Results")

        search_button.click(
            fn=search,
            inputs=[query_input, model_selector, refinement_toggle,
                    k1_slider, b_slider],
            outputs=[results_output, refinement_output],
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
