import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Final, Iterator, TypedDict

BASE_DIR: Final[str] = os.path.dirname(os.path.abspath(__file__))
SOURCE_DATA_DIR: Final[str] = os.path.join(BASE_DIR, "source_data")
ARTIFACTS_DIR: Final[str] = os.path.join(BASE_DIR, "artifacts")
XML_FILE: Final[str] = os.path.join(SOURCE_DATA_DIR, "honkai_star_rail_pages_current.xml")
OUTPUT_JSONL: Final[str] = os.path.join(ARTIFACTS_DIR, "hsr_v1_raw_lore.jsonl")
INSPECT_OUTPUT_DIR: Final[str] = os.path.join(BASE_DIR, "inspect_outputs")
DEBUG_OUTPUT_JSONL: Final[str] = f"{INSPECT_OUTPUT_DIR}/hsr_v1_debug_pages.jsonl"
LIMITED_OUTPUT_JSONL: Final[str] = f"{INSPECT_OUTPUT_DIR}/hsr_v1_raw_lore_sample.jsonl"

BANNED_TITLES = [
    "MediaWiki:", "Template:", "Category:", "User:", "File:", "Module:",
    "Talk:", "Guide", "Update/", "Version/", "Tier List", "/Media", "/Gallery"
]

BANNED_KEYWORDS = [
    "Photography Contest", "Web Event", "Twitch Drops", "HoYoLAB Community",
    "Submission Event", "Event Rewards", "Fujifilm", "Physical Rewards", "Official Release Trailer"
]

NOISY_SECTION_TITLES = {
    "Combat Info",
    "Ascensions and Stats",
    "Abilities",
    "Traces",
    "Eidolons",
    "Achievements",
    "Availability",
    "Event Warps",
    "Other Languages",
    "Change History",
    "References",
    "Navigation",
}

STRIP_PATTERNS = re.compile(
    r"(\|zh\s*=\s*.*?$)|"
    r"(\|zh_rm\s*=\s*.*?$)|"
    r"('{2,3})|"
    r"(\|nogroup=.*?$)|"
    r"(\|marker=.*?$)",
    re.MULTILINE | re.IGNORECASE
)


class LorePage(TypedDict):
    title: str
    raw_content: str
    cleaned_content: str


def ensure_parent_dir(path: str) -> None:
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def resolve_extract_output_path(limit: int | None, output_path: str | None) -> str:
    if output_path:
        return output_path

    if limit is not None:
        return LIMITED_OUTPUT_JSONL

    return OUTPUT_JSONL


def resolve_debug_output_path(output_path: str | None) -> str:
    return output_path or DEBUG_OUTPUT_JSONL


def page_matches_query(page: LorePage, query: str, scope: str) -> bool:
    normalized_query = query.lower()

    if scope == "title":
        haystacks = (page["title"],)
    elif scope == "content":
        haystacks = (page["cleaned_content"],)
    elif scope == "raw":
        haystacks = (page["raw_content"],)
    else:
        haystacks = (page["title"], page["cleaned_content"], page["raw_content"])

    return any(normalized_query in haystack.lower() for haystack in haystacks)


def extract_description_templates(text: str) -> str:
    """Extract Description template content and preserve it as plain text."""
    descriptions: list[str] = []
    
    # Match {{Description|...}} templates (handles nested content)
    pattern = r'\{\{Description\s*\|\s*([^}]*(?:\}(?!\}))?[^}]*)\}\}'
    for match in re.finditer(pattern, text, re.IGNORECASE):
        desc = match.group(1).strip()
        if desc:
            descriptions.append(desc)
    
    if descriptions:
        # Prepend descriptions to preserve key info
        text = '\n\n'.join(descriptions) + '\n\n' + text
    
    return text


def strip_nested_templates(text: str) -> str:
    previous = ""
    while text != previous:
        previous = text
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)
    return text


def drop_noisy_sections(text: str) -> str:
    cleaned_lines: list[str] = []
    skip_section = False

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if line.endswith(':'):
            section_name = line[:-1].strip()
            skip_section = section_name in NOISY_SECTION_TITLES
            if skip_section:
                continue

        if skip_section:
            continue

        cleaned_lines.append(raw_line)

    return '\n'.join(cleaned_lines)


def should_skip_title(title: str) -> bool:
    return any(banned in title for banned in BANNED_TITLES)


def should_skip_content(title: str, cleaned_text: str) -> bool:
    return any(banned_kw in title or banned_kw in cleaned_text for banned_kw in BANNED_KEYWORDS)


def clean_wikitext(text: str | None, title: str) -> str:
    if not text:
        return ""

    source_text: str = text

    low_title = title.lower()
    if any(suffix in low_title for suffix in ["/media", "/gallery", "/audio", "trivia", "/voice-overs"]):
        return ""

    lines: list[str] = source_text.split('\n')
    cleaned_lines: list[str] = []
    for line in lines:
        if line.count('|') > 2 and len(line) < 200:
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'<gallery>.*?</gallery>', '', text, flags=re.DOTALL | re.IGNORECASE)

    text = re.sub(r'==+\s*([^=]+?)\s*==+', r'\1:', text)
    text = drop_noisy_sections(text)

    text = re.sub(STRIP_PATTERNS, '', text)
    text = re.sub(r'^\s*\|.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\w+\s*=\s*.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'(?i)[\w\s-]+\.(png|jpg|jpeg|gif|webm|mp4)(\s*\|.*)?', '', text)
    text = re.sub(r'^\s*\[\[[a-z-]+:[^\]]+\]\]\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    text = extract_description_templates(text)
    
    text = strip_nested_templates(text)
    text = re.sub(r'^\s*\{\{.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\{\{[^\|\}]+\|?', '', text)
    text = re.sub(r'\}\}', '', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'(?i).*?\.ogg\s*', '', text)
    text = re.sub(r'^[:\s]*VO\s+.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^[:\s]*Arrow\s+', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = "".join(char for char in text if ord(char) < 128)
    text = re.sub(r'[\{\}\[\]]', '', text)
    text = re.sub(r'^\s*:\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\*+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    text = re.sub(r' +', ' ', text).strip()

    return text


def iter_lore_pages(query: str | None = None, query_scope: str = "title") -> Iterator[LorePage]:

    try:
        context = ET.iterparse(XML_FILE, events=('end',))
        _, root = next(context)

        for _, elem in context:
            if not elem.tag.endswith('page'):
                continue

            title_elem = elem.find('.//{*}title')
            text_elem = elem.find('.//{*}text')

            if title_elem is None or text_elem is None:
                elem.clear()
                root.clear()
                continue

            raw_title = title_elem.text
            raw_text = text_elem.text
            title = raw_title if isinstance(raw_title, str) else ""
            text = raw_text if isinstance(raw_text, str) else ""

            if should_skip_title(title):
                elem.clear()
                root.clear()
                continue

            cleaned_text = clean_wikitext(text, title)
            if should_skip_content(title, cleaned_text) or len(cleaned_text) <= 100:
                elem.clear()
                root.clear()
                continue

            page: LorePage = {
                "title": title,
                "raw_content": text,
                "cleaned_content": cleaned_text,
            }

            if query and not page_matches_query(page, query, query_scope):
                elem.clear()
                root.clear()
                continue

            yield page

            elem.clear()
            root.clear()
    except FileNotFoundError:
        raise SystemExit(f"XML dump not found: {XML_FILE}") from None
    except ET.ParseError as exc:
        line, column = exc.position
        raise SystemExit(
            f"Malformed XML dump: {XML_FILE} (line {line}, column {column}). "
            "The file may be truncated or corrupted."
        ) from None
    except OSError as exc:
        raise SystemExit(f"Unable to read XML dump {XML_FILE}: {exc}") from None


def write_clean_lore_jsonl(limit: int | None = None, output_path: str | None = None) -> None:
    print("Initializing streaming parser... (Grab a coffee, this takes < 60 seconds)")

    resolved_output_path = resolve_extract_output_path(limit=limit, output_path=output_path)
    ensure_parent_dir(resolved_output_path)

    saved_count = 0
    with open(resolved_output_path, 'w', encoding='utf-8') as output_file:
        for page in iter_lore_pages():
            data_point = {
                "title": page["title"],
                "content": page["cleaned_content"],
            }
            output_file.write(json.dumps(data_point, ensure_ascii=False) + "\n")
            saved_count += 1

            if saved_count % 5000 == 0:
                print(f"Saved {saved_count} valid lore targets.")

            if limit is not None and saved_count >= limit:
                break

    print(f"Done! Saved {saved_count} clean lore files to {resolved_output_path}")


def write_debug_pages(
    query: str,
    limit: int,
    output_path: str | None = None,
    query_scope: str = "title",
) -> None:
    resolved_output_path = resolve_debug_output_path(output_path)
    ensure_parent_dir(resolved_output_path)

    print(f"Exporting up to {limit} matching lore pages for {query_scope} query: {query}")

    saved_count = 0
    with open(resolved_output_path, 'w', encoding='utf-8') as output_file:
        for page in iter_lore_pages(query=query, query_scope=query_scope):
            debug_record = {
                "title": page["title"],
                "raw_content": page["raw_content"],
                "cleaned_content": page["cleaned_content"],
            }
            output_file.write(json.dumps(debug_record, ensure_ascii=False) + "\n")
            saved_count += 1

            if saved_count >= limit:
                break

    print(f"Done! Saved {saved_count} matching debug pages to {resolved_output_path}")