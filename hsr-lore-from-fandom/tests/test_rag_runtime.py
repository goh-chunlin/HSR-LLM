import pytest

from rag_runtime import RuntimeState


def test_runtime_state_should_eager_initialize_mode_handling() -> None:
    eager = RuntimeState(runtime_init_mode="eager")
    lazy = RuntimeState(runtime_init_mode="lazy")
    invalid = RuntimeState(runtime_init_mode="invalid")

    assert eager.should_eager_initialize() is True
    assert lazy.should_eager_initialize() is False
    assert invalid.should_eager_initialize() is False


def test_maybe_initialize_at_launch_calls_initialize_when_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = RuntimeState(runtime_init_mode="eager")
    called = {"count": 0}

    def _fake_initialize() -> None:
        called["count"] += 1

    monkeypatch.setattr(runtime, "initialize", _fake_initialize)
    runtime.maybe_initialize_at_launch()
    assert called["count"] == 1


def test_maybe_initialize_at_launch_skips_when_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = RuntimeState(runtime_init_mode="lazy")
    called = {"count": 0}

    def _fake_initialize() -> None:
        called["count"] += 1

    monkeypatch.setattr(runtime, "initialize", _fake_initialize)
    runtime.maybe_initialize_at_launch()
    assert called["count"] == 0


def test_initialize_returns_early_if_runtime_already_ready() -> None:
    runtime = RuntimeState()
    runtime.runtime_ready = True
    runtime.initialize()
    assert runtime.init_error is None


def test_initialize_returns_early_if_init_error_already_set() -> None:
    runtime = RuntimeState()
    runtime.init_error = "already failed"
    runtime.initialize()
    assert runtime.init_error == "already failed"
