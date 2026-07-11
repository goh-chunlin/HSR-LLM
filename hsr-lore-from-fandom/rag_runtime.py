import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, cast
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

from rag_types import LoreChunk


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DEFAULT_OVERLAY_PATH = os.path.join(ARTIFACTS_DIR, "hsr_v1_overlay.json")
DEFAULT_OVERLAY_TIMEOUT_SEC = 8.0


def _resolve_overlay_path() -> str:
    overlay_path = os.getenv("HSR_LORE_OVERLAY_PATH", DEFAULT_OVERLAY_PATH).strip()
    return overlay_path or DEFAULT_OVERLAY_PATH


def _resolve_overlay_url() -> str:
    return os.getenv("HSR_LORE_OVERLAY_URL", "").strip()


def _resolve_overlay_access_key() -> str:
    return os.getenv("HSR_LORE_OVERLAY_ACCESS_KEY", "").strip()


def _resolve_overlay_access_header_name() -> str:
    header_name = os.getenv("HSR_LORE_OVERLAY_ACCESS_HEADER", "x-access-key").strip()
    return header_name or "x-access-key"


def _resolve_overlay_timeout_sec() -> float:
    raw_value = os.getenv("HSR_LORE_OVERLAY_TIMEOUT_SEC", str(DEFAULT_OVERLAY_TIMEOUT_SEC)).strip()
    if not raw_value:
        return DEFAULT_OVERLAY_TIMEOUT_SEC

    try:
        timeout = float(raw_value)
        return timeout if timeout > 0 else DEFAULT_OVERLAY_TIMEOUT_SEC
    except ValueError:
        print(
            f"[STARTUP WARNING] Invalid HSR_LORE_OVERLAY_TIMEOUT_SEC={raw_value!r}; defaulting to {DEFAULT_OVERLAY_TIMEOUT_SEC}.",
            flush=True,
        )
        return DEFAULT_OVERLAY_TIMEOUT_SEC


def _empty_lore_chunks() -> list[LoreChunk]:
    return []


def _normalize_chunk_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", title.lower())).strip()


def _normalize_lore_chunk(item: object, source: str) -> LoreChunk:
    if isinstance(item, dict):
        item_dict = cast(dict[str, object], item)
        title = str(item_dict.get("title", ""))
        text = str(item_dict.get("text", ""))
    else:
        title = ""
        text = ""

    return cast(
        LoreChunk,
        {
            "title": title,
            "text": text,
            "source": source,
            "chunk_key": _normalize_chunk_title(title),
        },
    )


def _extract_overlay_records(payload: object) -> list[object]:
    if isinstance(payload, list):
        return cast(list[object], payload)

    if isinstance(payload, dict):
        payload_dict = cast(dict[str, object], payload)
        record = payload_dict.get("record")
        if isinstance(record, list):
            return cast(list[object], record)

        if isinstance(record, dict):
            return [cast(object, record)]

    raise RuntimeError("Invalid overlay metadata format: expected list or JSONBin record payload")


def _load_overlay_payload_from_file(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_overlay_payload_from_url(url: str, access_key: str, access_header_name: str, timeout_sec: float) -> object:
    headers = {"Accept": "application/json"}
    if access_key:
        headers[access_header_name] = access_key

    req = Request(url=url, headers=headers, method="GET")
    with urlopen(req, timeout=timeout_sec) as response:  # nosec B310 - URL is controlled by deployment env
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read().decode(charset, errors="replace")
    return json.loads(body)


def _merge_overlay_chunks(base_chunks: list[LoreChunk], overlay_chunks: list[LoreChunk]) -> list[LoreChunk]:
    merged_chunks = [
        cast(
            LoreChunk,
            {
                "title": str(chunk.get("title", "")),
                "text": str(chunk.get("text", "")),
                "source": str(chunk.get("source", "base")),
                "chunk_key": str(chunk.get("chunk_key", _normalize_chunk_title(str(chunk.get("title", ""))))),
            },
        )
        for chunk in base_chunks
    ]

    key_to_index = {
        str(chunk.get("chunk_key", _normalize_chunk_title(str(chunk.get("title", ""))))): index
        for index, chunk in enumerate(merged_chunks)
    }

    for chunk in overlay_chunks:
        key = str(chunk.get("chunk_key", _normalize_chunk_title(str(chunk.get("title", "")))))
        normalized_chunk = cast(
            LoreChunk,
            {
                "title": str(chunk.get("title", "")),
                "text": str(chunk.get("text", "")),
                "source": "overlay",
                "chunk_key": key,
            },
        )

        existing_index = key_to_index.get(key)
        if existing_index is None:
            key_to_index[key] = len(merged_chunks)
            merged_chunks.append(normalized_chunk)
        else:
            merged_chunks[existing_index] = normalized_chunk

    return merged_chunks


@dataclass
class RuntimeState:
    index_path: str = os.path.join(ARTIFACTS_DIR, "my_hsr_1.0_index.faiss")
    chunks_path: str = os.path.join(ARTIFACTS_DIR, "hsr_v1_chunks.json")
    overlay_path: str = field(default_factory=_resolve_overlay_path)
    overlay_url: str = field(default_factory=_resolve_overlay_url)
    overlay_access_key: str = field(default_factory=_resolve_overlay_access_key)
    overlay_access_header_name: str = field(default_factory=_resolve_overlay_access_header_name)
    overlay_timeout_sec: float = field(default_factory=_resolve_overlay_timeout_sec)
    runtime_init_mode: str = field(
        default_factory=lambda: os.getenv("HSR_RUNTIME_INIT_MODE", "lazy").strip().lower()
    )

    init_error: str | None = None
    embed_model: Any | None = None
    index: Any | None = None
    text_metadata: list[LoreChunk] = field(default_factory=_empty_lore_chunks)
    bm25: Any | None = None
    runtime_ready: bool = False
    overlay_record_count: int = 0
    overlay_replacement_count: int = 0
    overlay_source: str = "none"

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
                normalized_chunks.append(_normalize_lore_chunk(item, "base"))

            overlay_chunks: list[LoreChunk] = []
            overlay_loaded_chunks: list[object] = []

            if self.overlay_url:
                try:
                    print(f"Fetching overlay JSON metadata from {self.overlay_url}...", flush=True)
                    overlay_payload = _load_overlay_payload_from_url(
                        url=self.overlay_url,
                        access_key=self.overlay_access_key,
                        access_header_name=self.overlay_access_header_name,
                        timeout_sec=self.overlay_timeout_sec,
                    )
                    overlay_loaded_chunks = _extract_overlay_records(overlay_payload)
                    self.overlay_source = "remote"
                except (URLError, HTTPError, json.JSONDecodeError, RuntimeError, ValueError) as e:
                    print(f"[STARTUP WARNING] Remote overlay fetch failed: {e}", flush=True)
                    print("[STARTUP WARNING] Falling back to local overlay file when available.", flush=True)

            if not overlay_loaded_chunks:
                if os.path.isfile(self.overlay_path):
                    try:
                        print(f"Reading overlay JSON metadata chunks from {self.overlay_path}...", flush=True)
                        overlay_payload = _load_overlay_payload_from_file(self.overlay_path)
                        overlay_loaded_chunks = _extract_overlay_records(overlay_payload)
                        self.overlay_source = "local"
                    except (OSError, json.JSONDecodeError, RuntimeError, ValueError) as e:
                        print(f"[STARTUP WARNING] Local overlay load failed: {e}", flush=True)
                        print("[STARTUP WARNING] Continuing with base corpus only.", flush=True)
                        self.overlay_source = "none"
                else:
                    print(f"Overlay JSON not found at {self.overlay_path}; using base corpus only.", flush=True)
                    self.overlay_source = "none"

            for item in overlay_loaded_chunks:
                overlay_chunks.append(_normalize_lore_chunk(item, "overlay"))

            merged_chunks = _merge_overlay_chunks(normalized_chunks, overlay_chunks)
            self.overlay_record_count = len(overlay_chunks)
            self.overlay_replacement_count = max(0, len(normalized_chunks) + len(overlay_chunks) - len(merged_chunks))
            self.text_metadata = merged_chunks
            print(f"-> Metadata contains {len(self.text_metadata)} records.", flush=True)
            if self.overlay_record_count > 0:
                print(
                    f"-> Overlay ({self.overlay_source}) contributes {self.overlay_record_count} records and replaced {self.overlay_replacement_count} base records.",
                    flush=True,
                )

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
