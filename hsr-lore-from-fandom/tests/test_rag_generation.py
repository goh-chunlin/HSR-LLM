import sys
import types
from types import TracebackType
from typing import Any

import pytest

from rag_generation import generate_answer


class _FakeSpan:
    def __enter__(self) -> "_FakeSpan":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def set_attribute(self, _k: str, _v: object) -> None:
        return None

    def record_exception(self, _e: BaseException) -> None:
        return None


class _FakeTracer:
    def start_as_current_span(self, _name: str) -> _FakeSpan:
        return _FakeSpan()


def test_generate_answer_returns_error_message_on_client_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingClient:
        def __init__(self, _model: str) -> None:
            raise RuntimeError("hf down")

    fake_hf: Any = types.SimpleNamespace(InferenceClient=_FailingClient)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    result = generate_answer("Who is Kafka?", [], tracer=None, intent_label="other")
    assert result == "Error generating answer: hf down"


def test_generate_answer_strips_markup_from_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Msg:
        content: str = "<source_document><title>x</title><content>Final answer</content></source_document>"

    class _Choice:
        message: _Msg = _Msg()

    class _Response:
        choices: list[_Choice] = [_Choice()]

    class _Client:
        def __init__(self, _model: str) -> None:
            return None

        def chat_completion(self, **_kwargs: object) -> _Response:
            return _Response()

    fake_hf: Any = types.SimpleNamespace(InferenceClient=_Client)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    result = generate_answer("Who is Kafka?", [], tracer=_FakeTracer(), intent_label="entity_lookup")
    assert "<title>" not in result
    assert "<content>" not in result
    assert "Final answer" in result
