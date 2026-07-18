import importlib
import sys
import types
from types import ModuleType, TracebackType

import pytest

from rag_types import IntentClassification, IntentType, RetrievedChunk


class _FakeSpan:
    def __init__(self) -> None:
        self.attrs: dict[str, object] = {}
        self.exceptions: list[Exception] = []

    def __enter__(self) -> "_FakeSpan":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def set_attribute(self, key: str, value: object) -> None:
        self.attrs[key] = value

    def record_exception(self, exc: Exception) -> None:
        self.exceptions.append(exc)


class _FakeTracer:
    def start_as_current_span(self, _name: str) -> _FakeSpan:
        return _FakeSpan()


class _FakeCounter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, str]]] = []

    def add(self, value: int, attrs: dict[str, str]) -> None:
        self.calls.append((value, attrs))


class _FakeHistogram:
    def __init__(self) -> None:
        self.calls: list[tuple[float, dict[str, str]]] = []

    def record(self, value: float, attrs: dict[str, str]) -> None:
        self.calls.append((value, attrs))


class _FakeRuntime:
    def __init__(self, init_error: str | None = None) -> None:
        self.init_error = init_error
        self.initialize_calls = 0

    def initialize(self) -> None:
        self.initialize_calls += 1


def _fingerprint_text(value: str) -> str:
    return f"fp:{len(value)}"


def _normalize_empty(_query: str) -> str:
    _ = _query
    return ""


def _normalize_kafka(_query: str) -> str:
    _ = _query
    return "who is kafka"


def _classify_entity(_query: str) -> IntentClassification:
    _ = _query
    return {"label": "entity_lookup", "confidence": 0.9}


def _top_k_passthrough(_intent: IntentType, default_top_k: int) -> int:
    _ = (_intent, default_top_k)
    return 4


def _single_kafka_match(
    _query: str,
    runtime: object,
    top_k: int,
    intent_label: IntentType,
) -> list[RetrievedChunk]:
    _ = _query
    _ = (runtime, top_k, intent_label)
    return [{"title": "Kafka", "text": "A hunter", "score": 0.99}]


def _answer_kafka(
    _query: str,
    _retrieved_chunks: list[RetrievedChunk],
    tracer: object,
    intent_label: IntentType,
) -> str:
    _ = (_query, _retrieved_chunks, tracer, intent_label)
    return "Kafka is a Stellaron Hunter."


def _load_rag_service_with_fake_observability(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, _FakeCounter, _FakeHistogram, _FakeHistogram]:
    counter = _FakeCounter()
    latency_hist = _FakeHistogram()
    answer_hist = _FakeHistogram()

    fake_observability = types.SimpleNamespace(
        OTEL_CAPTURE_CONTENT=False,
        answer_chars_hist=answer_hist,
        fingerprint_text=_fingerprint_text,
        request_latency_ms_hist=latency_hist,
        requests_counter=counter,
        tracer=_FakeTracer(),
    )

    monkeypatch.setitem(sys.modules, "observability", fake_observability)
    sys.modules.pop("rag_service", None)
    module = importlib.import_module("rag_service")
    return module, counter, latency_hist, answer_hist


def test_hsr_rag_interface_handles_empty_query(monkeypatch: pytest.MonkeyPatch) -> None:
    rag_service, counter, _latency, answer_hist = _load_rag_service_with_fake_observability(monkeypatch)
    _ = _latency

    monkeypatch.setattr(rag_service, "normalize_user_query", _normalize_empty)

    runtime = _FakeRuntime()
    result = str(rag_service.hsr_rag_interface("   ", runtime))

    assert runtime.initialize_calls == 1
    assert result == "### Please enter a lore question."
    assert counter.calls[-1][1]["status"] == "empty_query"
    assert answer_hist.calls == []


def test_hsr_rag_interface_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    rag_service, counter, _latency, answer_hist = _load_rag_service_with_fake_observability(monkeypatch)
    _ = _latency

    monkeypatch.setattr(rag_service, "normalize_user_query", _normalize_kafka)
    monkeypatch.setattr(rag_service, "classify_query_intent", _classify_entity)
    monkeypatch.setattr(rag_service, "retrieval_top_k_for_intent", _top_k_passthrough)
    monkeypatch.setattr(rag_service, "retrieve_lore_hybrid", _single_kafka_match)
    monkeypatch.setattr(rag_service, "generate_answer", _answer_kafka)

    runtime = _FakeRuntime()
    result = str(rag_service.hsr_rag_interface("Who is Kafka?", runtime))

    assert runtime.initialize_calls == 1
    assert "## 💬 Answer" in result
    assert "Kafka is a Stellaron Hunter." in result
    assert "- **Kafka** (Score: 0.9900)" in result
    assert counter.calls[-1][1]["status"] == "ok"
    assert len(answer_hist.calls) == 1
