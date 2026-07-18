from rag_runtime import RuntimeState
from rag_retrieval import normalize_user_query, retrieve_lore_hybrid, tokenize_text


def test_normalize_user_query_collapses_whitespace_and_newlines() -> None:
    query = "  Who\tis\r\n\r\n\r\nDan Heng?   "
    assert normalize_user_query(query) == "Who is\n\nDan Heng?"


def test_normalize_user_query_applies_length_limit() -> None:
    query = "a" * 20
    assert normalize_user_query(query, max_chars=10) == "a" * 10


def test_tokenize_text_returns_non_empty_tokens() -> None:
    tokens = tokenize_text("Silver Wolf is hacking systems")
    assert len(tokens) > 0
    assert any("silver" in token for token in tokens)


def test_retrieve_lore_hybrid_returns_empty_when_runtime_not_ready() -> None:
    runtime = RuntimeState()
    assert retrieve_lore_hybrid("Who is Kafka?", runtime) == []
