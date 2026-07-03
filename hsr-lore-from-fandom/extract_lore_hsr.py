import xml.etree.ElementTree as ET
import json
import re
from typing import Final

# UPDATE THIS to match your exact downloaded filename
XML_FILE: Final[str] = "honkai_star_rail_pages_current.xml"
OUTPUT_JSONL: Final[str] = "hsr_v1_raw_lore.jsonl"

# Filter out trash pages we do not want to train on
BANNED_TITLES = [
    "MediaWiki:", "Template:", "Category:", "User:", "File:", "Module:", 
    "Talk:", "Guide", "Update/", "Version/", "Tier List", "/Media", "/Gallery"
]

# Real-world out-of-game event markers
BANNED_KEYWORDS = [
    "Photography Contest", "Web Event", "Twitch Drops", "HoYoLAB Community", 
    "Submission Event", "Event Rewards", "Fujifilm", "Physical Rewards", "Official Release Trailer"
]

STRIP_PATTERNS = re.compile(
    r"(\|zh\s*=\s*.*?$)|"       # Matches |zh= and everything after it on that line
    r"(\|zh_rm\s*=\s*.*?$)|"    # Matches |zh_rm= and everything after it on that line
    r"('{2,3})|"                # Matches triple (''') or double ('') wiki quotes
    r"(\|nogroup=.*?$)|"        # Matches leftovers like |nogroup=1
    r"(\|marker=.*?$)",         # Matches leftovers like |marker=
    re.MULTILINE | re.IGNORECASE
)

def clean_wikitext(text: str | None, title: str) -> str:
    """Smarter cleaner to isolate true narrative prose and dialogue."""
    if not text:
        return ""

    source_text: str = text
        
    low_title = title.lower()
    if any(suffix in low_title for suffix in ["/media", "/gallery", "/audio", "trivia", "/voice-overs"]):
        return ""

    # 1. Strip lines that are just collections of tags separated by pipes (e.g., puppet|true form|the enemy)
    lines: list[str] = source_text.split('\n')
    cleaned_lines: list[str] = []
    for line in lines:
        # If a line contains multiple pipes, it's likely a metadata/tag line rather than prose
        if line.count('|') > 2 and len(line) < 200:
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # 2. Convert Wiki Headers (== Title ==) into clean structural section titles
    text = re.sub(r'==+\s*([^=]+?)\s*==+', r'\1:', text)

    # 3. Clean translation tags and wiki quotes (''')
    text = re.sub(STRIP_PATTERNS, '', text)

    # 4. Strip template metadata rows (lines starting with | or variable assignments)
    text = re.sub(r'^\s*\|.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\w+\s*=\s*.*$', '', text, flags=re.MULTILINE)

    # 5. Strip URLs entirely
    text = re.sub(r'https?://\S+', '', text)
    
    # 6. Strip standard wiki image/file syntax lines
    text = re.sub(r'(?i)[\w\s-]+\.(png|jpg|jpeg|gif|webm|mp4)(\s*\|.*)?', '', text)
    
    # 7. Clean legacy brackets and markup tags
    text = re.sub(r'\{\{[^\|\}]+\|?', '', text)
    text = re.sub(r'\}\}', '', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'(?i).*?\.ogg\s*', '', text)
    text = re.sub(r'^[:\s]*VO\s+.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^[:\s]*Arrow\s+', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # 8. Filter strictly for English printable text (ASCII)
    text = "".join(char for char in text if ord(char) < 128)
    
    # 9. Clean up layout architecture
    text = re.sub(r'[\{\}\[\]]', '', text) # Clear remaining random brackets
    
    # 10. Standardize newline separation
    # Collapse 3 or more consecutive newlines down to exactly two (preserving paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text).strip()
    
    return text

def parse_wiki_dump() -> None:
    print("Initializing streaming parser... (Grab a coffee, this takes < 60 seconds)")
    
    count = 0
    saved_count = 0
    
    # Using 'iterparse' ensures we do not load 5M lines into memory at once
    try:
        context = ET.iterparse(XML_FILE, events=('end',))

        # Strip namespaces automatically
        _, root = next(context)

        with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
            for _, elem in context:
                # Look for the end of a </page> block
                if elem.tag.endswith('page'):
                    title_elem = elem.find('.//{*}title')
                    text_elem = elem.find('.//{*}text')
                    
                    if title_elem is not None and text_elem is not None:
                        raw_title = title_elem.text
                        raw_text = text_elem.text
                        title = raw_title if isinstance(raw_title, str) else ""
                        text = raw_text if isinstance(raw_text, str) else ""
                        
                        # 1. Skip if it is a system/meta/wiki page
                        if any(banned in title for banned in BANNED_TITLES):
                            elem.clear()
                            root.clear()
                            continue

                        cleaned_text = clean_wikitext(text, title)

                        if any(banned_kw in title or banned_kw in cleaned_text for banned_kw in BANNED_KEYWORDS):
                            elem.clear()
                            root.clear()
                            continue
                            
                        if len(cleaned_text) > 100:
                            data_point = {
                                "title": title,
                                "content": cleaned_text
                            }
                            f.write(json.dumps(data_point, ensure_ascii=False) + "\n")
                            saved_count += 1
                    
                    count += 1
                    if count % 5000 == 0:
                        print(f"Processed {count} pages... Saved {saved_count} valid lore targets.")
                    
                    # Critical Memory Management: clear the node from RAM
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

    print(f"Done! Saved {saved_count} clean lore files to {OUTPUT_JSONL}")

if __name__ == "__main__":
    parse_wiki_dump()