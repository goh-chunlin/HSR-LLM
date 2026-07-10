import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

print("=== APP BOOT: pre-import ===", flush=True)

print("=== APP BOOT: importing gradio ===", flush=True)
import gradio as ui
print("=== APP BOOT: gradio imported ===", flush=True)
from rag_retrieval import MAX_USER_QUERY_CHARS
from rag_runtime import RuntimeState
from rag_service import hsr_rag_interface

print("=== APP MODULE IMPORT START ===", flush=True)

runtime = RuntimeState()


def _run_query(user_query: str) -> str:
    return hsr_rag_interface(user_query, runtime)

# Build the simple Gradio layout
demo = ui.Interface(
    fn=_run_query,
    inputs=ui.Textbox(
        label="Ask a Honkai: Star Rail Lore Question",
        placeholder="Who is Member 83 in Genius Society?",
        max_length=MAX_USER_QUERY_CHARS,
        info=f"Input is limited to {MAX_USER_QUERY_CHARS} characters.",
    ),
    outputs=ui.Markdown(),
    title="🌌 Honkai: Star Rail Lore RAG Engine",
    description=(
        "Hybrid retrieval backend combining BM25 keyword matching and FAISS dense vector embeddings. "
    )
)

print("=== GRADIO INTERFACE READY ===", flush=True)

if __name__ == "__main__":
    # Hugging Face Spaces looks for a running web server on port 7860 by default
    print("=== LAUNCHING GRADIO APP ===", flush=True)
    runtime.maybe_initialize_at_launch()
    demo.launch(theme="soft")
