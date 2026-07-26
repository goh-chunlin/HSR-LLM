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
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class _FailingClient:
        def __init__(self) -> None:
            raise RuntimeError("groq down")

    class _ChatModule:
        ChatCompletion = object

    fake_groq: Any = types.SimpleNamespace(Groq=_FailingClient)
    monkeypatch.setitem(sys.modules, "groq", fake_groq)
    monkeypatch.setitem(sys.modules, "groq.types.chat", _ChatModule)

    result = generate_answer("Who is Kafka?", [], tracer=None, intent_label="other")
    assert result == "Error generating answer: groq down"


def test_generate_answer_strips_markup_from_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class _Msg:
        content: str = "<source_document><title>x</title><content>Final answer</content></source_document>"

    class _Choice:
        message: _Msg = _Msg()

    class _Response:
        choices: list[_Choice] = [_Choice()]

    class _Completions:
        def create(self, **_kwargs: object) -> _Response:
            return _Response()

    class _Chat:
        completions: _Completions = _Completions()

    class _Client:
        def __init__(self) -> None:
            return None

        chat: _Chat = _Chat()

    class _ChatModule:
        ChatCompletion = _Response

    fake_groq: Any = types.SimpleNamespace(Groq=_Client)
    monkeypatch.setitem(sys.modules, "groq", fake_groq)
    monkeypatch.setitem(sys.modules, "groq.types.chat", _ChatModule)

    result = generate_answer("Who is Kafka?", [], tracer=_FakeTracer(), intent_label="entity_lookup")
    assert "<title>" not in result
    assert "<content>" not in result
    assert "Final answer" in result
