import builtins
import io
import json
import runpy
import sys
import types
from pathlib import Path
from typing import IO

import pytest


def _fake_json_load(_file_obj: IO[str]) -> list[dict[str, str]]:
    return [{"title": "Doc", "text": "contains 83"}]


def _fake_open(*_args: object, **_kwargs: object) -> IO[str]:
    return io.StringIO("[]")


def test_diagnose_script_runs_with_stubbed_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    diagnose_path = Path(__file__).resolve().parents[1] / "diagnose.py"

    class _FakeRuntime:
        def __init__(self) -> None:
            self.overlay_source = "none"
            self.overlay_path = "artifacts/overlay.json"
            self.overlay_record_count = 0
            self.overlay_replacement_count = 0
            self.text_metadata = []
            self.chunks_path = "artifacts/fake_chunks.json"

        def initialize(self) -> None:
            return None

    fake_runtime_module = types.SimpleNamespace(RuntimeState=_FakeRuntime)
    monkeypatch.setitem(sys.modules, "rag_runtime", fake_runtime_module)

    monkeypatch.setattr(json, "load", _fake_json_load)
    monkeypatch.setattr(builtins, "open", _fake_open)

    runpy.run_path(str(diagnose_path), run_name="__main__")
    captured: str = capsys.readouterr().out

    assert "--- Runtime summary ---" in captured
    assert "Overlay source:" in captured
    assert "[Index 0] Source Document: Doc" in captured
