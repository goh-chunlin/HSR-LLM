from observability import fingerprint_text


def test_fingerprint_text_is_stable_and_sha256_length() -> None:
    one = fingerprint_text("hello world")
    two = fingerprint_text("hello world")
    assert one == two
    assert len(one) == 64


def test_fingerprint_text_changes_with_input() -> None:
    assert fingerprint_text("a") != fingerprint_text("b")
