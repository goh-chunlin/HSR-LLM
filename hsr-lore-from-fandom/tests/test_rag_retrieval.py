import sys
import types
from typing import Any, cast

import pytest

from rag_runtime import RuntimeState
from rag_retrieval import normalize_user_query, retrieve_lore_hybrid, tokenize_text


def test_normalize_user_query_collapses_whitespace_and_newlines() -> None:
    query = "  Who\tis\r\n\r\n\r\nDan Heng?   "
    assert normalize_user_query(query) == "Who is\n\nDan Heng?"


def test_normalize_user_query_applies_length_limit() -> None:
    query = "a" * 20
    assert normalize_user_query(query, max_chars=10) == "a" * 10


def test_tokenize_text_returns_non_empty_tokens() -> None:
    tokens = tokenize_text("Silver Wolf is hacking systems")
    assert len(tokens) > 0
    assert any("silver" in token for token in tokens)


def test_retrieve_lore_hybrid_returns_empty_when_runtime_not_ready() -> None:
    runtime = RuntimeState()
    assert retrieve_lore_hybrid("Who is Kafka?", runtime) == []


def test_retrieve_lore_hybrid_carries_optional_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeBm25:
        def get_scores(self, _query_tokens: list[str]) -> list[float]:
            return [1.0]

    class _FakeEmbedModel:
        def encode(self, _queries: list[str], convert_to_numpy: bool = True) -> list[list[float]]:
            _ = convert_to_numpy
            return [[1.0, 0.0]]

    class _FakeIndex:
        def search(self, _query_vector: list[list[float]], _k: int) -> tuple[list[list[float]], list[list[int]]]:
            return [[0.9]], [[0]]

    def _normalize_l2(_vector: object) -> None:
        return None

    fake_faiss = types.SimpleNamespace(normalize_L2=_normalize_l2)
    fake_np = types.SimpleNamespace(ndarray=list)
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    monkeypatch.setitem(sys.modules, "numpy", fake_np)

    runtime = RuntimeState()
    runtime.runtime_ready = True
    runtime.embed_model = _FakeEmbedModel()
    runtime.index = _FakeIndex()
    runtime.bm25 = _FakeBm25()
    runtime.text_metadata = cast(list[Any], [
        {
            "title": "Kafka",
            "text": "A Stellaron Hunter",
            "source": "overlay",
            "reference": {
                "sourceName": "HSR Wiki",
                "sourceUrl": "https://example.com/kafka",
            },
            "media": [
                {
                    "url": "https://example.com/kafka.png",
                    "type": "image",
                }
            ],
        }
    ])

    matches = retrieve_lore_hybrid("Who is Kafka?", runtime=runtime, top_k=1, intent_label="entity_lookup")

    assert len(matches) == 1
    assert matches[0]["title"] == "Kafka"
    reference = matches[0].get("reference")
    media = matches[0].get("media")
    assert reference is not None
    assert media is not None
    assert reference["sourceName"] == "HSR Wiki"
    assert media[0]["type"] == "image"
