import sys

import pytest

from extract_lore_hsr import parse_args


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["extract_lore_hsr.py"])
    args = parse_args()
    assert args.limit is None
    assert args.output is None


def test_parse_args_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["extract_lore_hsr.py", "--limit", "10", "--output", "artifacts/out.jsonl"],
    )
    args = parse_args()
    assert args.limit == 10
    assert args.output == "artifacts/out.jsonl"
