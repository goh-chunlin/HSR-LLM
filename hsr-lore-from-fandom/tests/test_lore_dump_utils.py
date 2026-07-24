from lore_dump_utils import (
    clean_wikitext,
    extract_description_templates,
    extract_gallery_media,
    get_fandom_image_url,
)


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


def test_get_fandom_image_url_builds_static_cdn_path() -> None:
    result = get_fandom_image_url("Character Asta Splash Art.png|Splash Art")
    assert result is not None
    assert result.startswith("https://static.wikia.nocookie.net/houkai-star-rail/images/")
    assert "Character_Asta_Splash_Art.png" in result
    assert result.endswith("/revision/latest")


def test_extract_gallery_media_parses_image_lines_and_caption_links() -> None:
    raw = """
==Character Introduction==
<gallery>
Character Asta Introduction.png|[https://www.hoyolab.com/article/4610963 Official Introduction]
Character Asta Splash Art.png|Splash Art
</gallery>
"""
    media = extract_gallery_media(raw)

    assert len(media) == 2
    assert media[0]["type"] == "image"
    assert media[0]["title"] == "Official Introduction"
    assert media[0]["attributionUrl"] == "https://www.hoyolab.com/article/4610963"
    assert media[1]["title"] == "Splash Art"
