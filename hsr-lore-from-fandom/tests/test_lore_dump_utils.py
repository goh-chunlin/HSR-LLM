from lore_dump_utils import clean_wikitext, extract_description_templates


def test_extract_description_templates_prepends_description_text() -> None:
    raw = "{{Description|master hacker}}\nSome other content"
    result = extract_description_templates(raw)
    assert result.startswith("master hacker")


def test_clean_wikitext_keeps_description_content() -> None:
    raw = "{{Description|master hacker}}\n'''Silver Wolf''' is a character in the game."
    cleaned = clean_wikitext(raw, "Silver Wolf")
    assert "master hacker" in cleaned.lower()


def test_clean_wikitext_drops_media_gallery_titles() -> None:
    raw = "This should not matter"
    cleaned = clean_wikitext(raw, "Silver Wolf/Media")
    assert cleaned == ""
