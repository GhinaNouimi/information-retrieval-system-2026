import gradio as gr

from services.retrieval_service.strategies.bm25_large_strategy import BM25LargeRetrievalStrategy
from services.retrieval_service.strategies.embedding_strategy import EmbeddingRetrievalStrategy
from services.retrieval_service.strategies.hybrid_serial_strategy import HybridSerialRetrievalStrategy
from services.retrieval_service.strategies.hybrid_parallel_strategy import HybridParallelRetrievalStrategy
from services.query_refinement_service.query_expander import refine_query


# تحميل النماذج مرة واحدة عند بدء التشغيل
print("Loading strategies... please wait.")
STRATEGIES = {
    "BM25": BM25LargeRetrievalStrategy(),
    "Embedding": EmbeddingRetrievalStrategy(),
    "Hybrid Serial": HybridSerialRetrievalStrategy(),
    "Hybrid Parallel": HybridParallelRetrievalStrategy(),
}
print("All strategies loaded.")


def search(query: str, model_name: str, use_refinement: bool):
    """
    الدالة الرئيسية التي تستدعيها الواجهة عند الضغط على Search.
    """
    if not query.strip():
        return "Please enter a query.", ""

    # تطبيق Query Refinement إذا كان مفعّلاً
    refined_info = ""
    final_query = query

    if use_refinement:
        refined = refine_query(query)
        final_query = refined["final"]

        if refined["corrected"] != refined["original"]:
            refined_info += f"**Spell Correction:** {refined['original']} → {refined['corrected']}\n\n"

        refined_info += f"**Expanded Query:** {refined['expanded']}"

    # تشغيل البحث
    strategy = STRATEGIES[model_name]
    results = strategy.search(final_query)

    # تنسيق النتائج
    output = ""
    for result in results:
        output += f"**Rank {result['rank']}** — Score: {result['score']:.4f}\n"
        output += f"Doc ID: {result['doc_id']}\n"
        output += f"{result['text']}\n"
        output += "---\n"

    return output, refined_info


def build_ui():
    with gr.Blocks(title="IR Search System") as app:

        gr.Markdown("# Information Retrieval Search System")
        gr.Markdown("Search through 50,000 documents using different retrieval models.")

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Enter your query",
                    placeholder="e.g. how to invest in stock market",
                    lines=2,
                )
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    choices=["BM25", "Embedding", "Hybrid Serial", "Hybrid Parallel"],
                    value="BM25",
                    label="Retrieval Model",
                )
                refinement_toggle = gr.Checkbox(
                    label="Enable Query Refinement",
                    value=False,
                )
                search_button = gr.Button("Search", variant="primary")

        refinement_output = gr.Markdown(label="Query Refinement Info", visible=True)
        results_output = gr.Markdown(label="Search Results")

        search_button.click(
            fn=search,
            inputs=[query_input, model_selector, refinement_toggle],
            outputs=[results_output, refinement_output],
        )

        gr.Markdown("---")
        gr.Markdown("**Dataset:** beir/quora/test | **Documents indexed:** 50,000")

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch()