from types import SimpleNamespace

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
