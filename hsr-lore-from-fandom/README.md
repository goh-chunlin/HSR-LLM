---
title: HSR Lore RAG Engine
emoji: 🌌
colorFrom: purple
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

Ask any Honkai: Star Rail lore question and get a cited answer.

Powered by a hybrid BM25 + FAISS retrieval pipeline over 22k chunks from the HSR Fandom wiki, with answers synthesised by Llama 3.1 8B Instruct.

Expected result:

- A new `hsr_v1_raw_lore.jsonl` file in this folder.
- Progress logs every 5000 pages, then a final saved-count summary.

If your XML filename is different, update `XML_FILE` in `extract_lore_hsr.py` first.

## Files

- `extract_lore_hsr.py`: Main extraction script.
- `build_lore_vector_db.py`: Builds FAISS index and chunk metadata from extracted lore.
- `honkai_star_rail_pages_current.xml`: Input XML dump file.
- `hsr_v1_raw_lore.jsonl`: Output JSONL file (generated after running the script).

## Vector DB Build (FAISS)

Use this after generating `hsr_v1_raw_lore.jsonl`:

```bash
python3 build_lore_vector_db.py
```

Expected output files:

- `my_hsr_1.0_index.faiss`
- `hsr_v1_chunks.json`

## FAISS Stability Notes (macOS)

If you hit a segmentation fault while initializing FAISS, it is commonly related to native threading libraries.

This repo now applies a thread-limit workaround inside `build_lore_vector_db.py` before loading ML libraries:

- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `VECLIB_MAXIMUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`

You usually do not need to export these manually, because the script sets them at startup.

Python compatibility note:

- Python 3.13 is supported in this project setup.
- Python 3.14+ is blocked by the script guard due to known FAISS wheel instability on macOS.

## What the Script Does

`extract_lore_hsr.py`:

1. Streams the XML dump with `xml.etree.ElementTree.iterparse` (memory efficient for large files).
2. Skips non-content/system pages based on banned title prefixes/keywords.
3. Keeps pages that match lore-focused keywords in either:
   - page title, or
   - first ~2000 characters of page text.
4. Cleans common MediaWiki markup using regex.
5. Writes each valid page as one JSON line with:
   - `title`
   - `content`

## Current Filter Scope

### Included keyword anchors

- Herta Space Station
- Belobog
- Jarilo-VI
- Cocolia
- Bronya
- Seele
- Sampo
- Gepard
- Asta
- Arlan
- Simulated Universe

### Excluded by title patterns

- `MediaWiki:`
- `Template:`
- `Category:`
- `User:`
- `File:`
- `Module:`
- `Talk:`
- any title containing `Guide`, `Update/`, `Version/`, `Tier List`

## Requirements

- Python 3.9+ (standard library only)
- A valid MediaWiki XML dump file

No extra pip packages are required.

## Usage

From this directory (`hsr-lore-from-fandom`):

```bash
python3 extract_lore_hsr.py
```

If successful, you will see progress logs every 5000 pages and a final summary.

## Input and Output Configuration

Inside `extract_lore_hsr.py`, update these constants if needed:

```python
XML_FILE = "honkai_star_rail_pages_current.xml"
OUTPUT_JSONL = "hsr_v1_raw_lore.jsonl"
```

Use a different output filename if you want multiple extraction variants.

## Output Format (JSONL)

Each line is an independent JSON object:

```json
{"title": "Belobog", "content": "...cleaned lore text..."}
```

This format is convenient for later dataset processing and LoRA training pipelines.

## Notes and Limitations

- Cleanup is intentionally simple (regex-based), not a full wikitext parser.
- Some residual markup artifacts may remain depending on page complexity.
- Filtering is keyword-driven, so relevant pages without anchor terms may be missed.
- Conversely, some matched pages may still contain non-lore text and may need manual post-filtering.

## Suggested Next Step

After extraction, consider a second pass script to:

- deduplicate near-identical entries,
- split long articles into smaller chunks,
- add metadata tags (region, faction, character),
- remove remaining low-signal pages.
