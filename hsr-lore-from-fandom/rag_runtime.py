import json
import os
from dataclasses import dataclass, field
from typing import Any, cast

from rag_types import LoreChunk


def _empty_lore_chunks() -> list[LoreChunk]:
    return []


@dataclass
class RuntimeState:
    index_path: str = "my_hsr_1.0_index.faiss"
    chunks_path: str = "hsr_v1_chunks.json"
    runtime_init_mode: str = field(
        default_factory=lambda: os.getenv("HSR_RUNTIME_INIT_MODE", "lazy").strip().lower()
    )

    init_error: str | None = None
    embed_model: Any | None = None
    index: Any | None = None
    text_metadata: list[LoreChunk] = field(default_factory=_empty_lore_chunks)
    bm25: Any | None = None
    runtime_ready: bool = False

    def should_eager_initialize(self) -> bool:
        if self.runtime_init_mode in {"eager", "lazy"}:
            return self.runtime_init_mode == "eager"

        print(
            f"[STARTUP WARNING] Invalid HSR_RUNTIME_INIT_MODE={self.runtime_init_mode!r}; defaulting to lazy.",
            flush=True,
        )
        return False

    def initialize(self) -> None:
        if self.runtime_ready or self.init_error is not None:
            return

        try:
            import faiss
            from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
            from sentence_transformers import SentenceTransformer

            from rag_retrieval import tokenize_text

            print("=== DEBUGGING INITIALIZATION ===", flush=True)
            print("Loading sentence-transformers/all-MiniLM-L6-v2...", flush=True)
            self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

            if not os.path.isfile(self.index_path):
                raise FileNotFoundError(f"Missing required artifact: {self.index_path}")

            print("Reading FAISS Index...", flush=True)
            self.index = faiss.read_index(self.index_path)
            print(f"-> FAISS index contains {self.index.ntotal} vectors.", flush=True)

            if not os.path.isfile(self.chunks_path):
                raise FileNotFoundError(f"Missing required artifact: {self.chunks_path}")

            print("Reading JSON metadata chunks...", flush=True)
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                loaded_payload: object = json.load(f)

            if not isinstance(loaded_payload, list):
                raise RuntimeError(f"Invalid metadata format in {self.chunks_path}: expected list")

            loaded_chunks = cast(list[object], loaded_payload)

            normalized_chunks: list[LoreChunk] = []
            for item in loaded_chunks:
                if isinstance(item, dict):
                    item_dict = cast(dict[str, object], item)
                    normalized_chunks.append(
                        {
                            "title": str(item_dict.get("title", "")),
                            "text": str(item_dict.get("text", "")),
                        }
                    )

            self.text_metadata = normalized_chunks
            print(f"-> Metadata contains {len(self.text_metadata)} records.", flush=True)

            if self.index.ntotal != len(self.text_metadata):
                print(
                    f"[CRITICAL WARNING] Row count mismatch! FAISS ({self.index.ntotal}) != JSON ({len(self.text_metadata)})",
                    flush=True,
                )

            print("Tokenizing entire corpus for BM25...", flush=True)
            tokenized_corpus = [
                tokenize_text(f"{title} {text}")
                for chunk in self.text_metadata
                for title, text in [
                    (
                        chunk.get("title", ""),
                        chunk.get("text", ""),
                    )
                ]
            ]
            self.bm25 = BM25Okapi(tokenized_corpus, k1=2.0, b=0.75)
            print("-> BM25 Initialization complete.", flush=True)
            self.runtime_ready = True
        except Exception as e:
            self.init_error = str(e)
            print(f"[STARTUP ERROR] {self.init_error}", flush=True)

    def maybe_initialize_at_launch(self) -> None:
        if self.should_eager_initialize():
            print("=== EAGER RUNTIME INITIALIZATION ENABLED ===", flush=True)
            self.initialize()
