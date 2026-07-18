import pytest

from rag_runtime import RuntimeState, _normalize_lore_chunk


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


def test_normalize_lore_chunk_preserves_valid_metadata() -> None:
    chunk = _normalize_lore_chunk(
        {
            "title": "Silver Wolf",
            "text": "A master hacker.",
            "reference": {
                "sourceName": "HSR Wiki",
                "sourceUrl": "https://example.com/silver-wolf",
                "license": "CC-BY-SA-3.0",
            },
            "media": [
                {
                    "url": "https://example.com/silver-wolf.jpg",
                    "type": "image",
                    "title": "Silver Wolf Artwork",
                }
            ],
        },
        source="overlay",
    )

    assert chunk["source"] == "overlay"
    assert chunk["reference"]["sourceName"] == "HSR Wiki"
    assert chunk["media"][0]["type"] == "image"


def test_normalize_lore_chunk_drops_invalid_metadata() -> None:
    chunk = _normalize_lore_chunk(
        {
            "title": "Invalid Metadata",
            "text": "Test",
            "reference": {"sourceName": "Missing URL"},
            "media": [
                {"type": "image"},
                {"url": "https://example.com/clip.mp4", "type": "audio"},
            ],
        },
        source="overlay",
    )

    assert "reference" not in chunk
    assert "media" not in chunk
