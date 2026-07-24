from types import SimpleNamespace
from typing import Any

import build_lore_vector_db as db
import pytest


def test_chunk_text_uses_overlap() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = db.chunk_text(text, chunk_size=10, overlap=2)
    assert chunks[0] == "abcdefghij"
    assert chunks[1].startswith("ijkl")


def test_validate_runtime_too_old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db.sys, "version_info", SimpleNamespace(major=3, minor=9))
    assert db.validate_runtime() is False


def test_validate_runtime_too_new(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db.sys, "version_info", SimpleNamespace(major=3, minor=14))
    assert db.validate_runtime() is False


def test_validate_runtime_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db.sys, "version_info", SimpleNamespace(major=3, minor=12))
    assert db.validate_runtime() is True


def test_normalize_media_records_filters_invalid_entries() -> None:
    raw_media: list[Any] = [
        {"url": "https://example.com/a.png", "type": "image", "title": "A"},
        {"url": "https://example.com/b.mp4", "type": "video", "description": "B"},
        {"url": "", "type": "image"},
        {"url": "https://example.com/c.png", "type": "document"},
        "not-a-dict",
    ]

    normalized = db.normalize_media_records(raw_media)
    assert len(normalized) == 2
    assert normalized[0]["type"] == "image"
    assert normalized[1]["type"] == "video"


def test_normalize_media_records_handles_non_list() -> None:
    assert db.normalize_media_records(None) == []
