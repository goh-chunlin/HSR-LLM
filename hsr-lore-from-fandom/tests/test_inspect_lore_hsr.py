import sys

import pytest

from inspect_lore_hsr import parse_args


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["inspect_lore_hsr.py", "--query", "Kafka"])
    args = parse_args()
    assert args.query == "Kafka"
    assert args.query_scope == "content"
    assert args.limit == 5
    assert args.debug_output is None


def test_parse_args_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect_lore_hsr.py",
            "--query",
            "Stellaron Hunters",
            "--query-scope",
            "title",
            "--limit",
            "3",
            "--debug-output",
            "inspect_outputs/custom.jsonl",
        ],
    )
    args = parse_args()
    assert args.query == "Stellaron Hunters"
    assert args.query_scope == "title"
    assert args.limit == 3
    assert args.debug_output == "inspect_outputs/custom.jsonl"
