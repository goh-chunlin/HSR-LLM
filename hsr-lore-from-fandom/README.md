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

## Minimal Unit Tests

Use a small pytest baseline before adding features:

```bash
pip install -r requirements.txt
pytest -q
```

The current suite covers deterministic helpers (intent routing and text cleaning) and avoids heavy model/index initialization.

## Runtime Initialization

The app defaults to lazy initialization, which keeps startup faster but makes the first query pay the full model/index/BM25 warmup cost.

Set `HSR_RUNTIME_INIT_MODE=eager` to initialize the runtime at app launch instead. This is usually better when you want the first user request to respond faster.

Set `HSR_RUNTIME_INIT_MODE=lazy` to keep the current behavior.

## Runtime Overlay Corrections

If you have a small curated correction file, place it at `artifacts/hsr_v1_overlay.json` or point `HSR_LORE_OVERLAY_PATH` at an alternate JSON file.

Runtime now reads overlay metadata from local files only. If your source of truth is remote, sync it into this repo (for example via a scheduled GitHub Action) and let Spaces deploy the updated artifact.

The overlay file should be a JSON list of objects with at least `title` and `text`. When a title matches an existing base chunk, the overlay record replaces it in memory; otherwise it is appended as an additional record.

Overlay records receive an extra ranking boost controlled by `HSR_LORE_OVERLAY_SCORE_BONUS`.

```bash
export HSR_LORE_OVERLAY_PATH=artifacts/hsr_v1_overlay.json
export HSR_LORE_OVERLAY_SCORE_BONUS=0.4
```

Expected payload formats:
- Plain list: `[ {"title": "...", "text": "..."}, ... ]`
- Wrapped object: `{ "record": [ {"title": "...", "text": "..."}, ... ] }`

The default bias is intentionally stronger than a neutral tie-break so corrected lore can outrank incomplete base data more reliably.

## Smaller Debug Workflows

The XML wiki dump is large enough that it should not be your debugging unit. Use the extractor to create compact JSONL artifacts instead.

Default generated artifacts now live under `artifacts/`, and the raw XML dump lives under `source_data/`.

```bash
python3 extract_lore_hsr.py --limit 20
```

That writes only the first 20 cleaned lore pages so you can iterate without regenerating the full dataset.

If you omit `--output`, full extraction runs default to `artifacts/hsr_v1_raw_lore.jsonl`, while limited runs default to `inspect_outputs/hsr_v1_raw_lore_sample.jsonl`, which is gitignored for tester-friendly scratch output.

```bash
python3 inspect_lore_hsr.py --query Kafka --query-scope title --limit 3
```

That writes up to 3 matching pages with both `raw_content` and `cleaned_content`, which is the fastest way to debug why a chatbot answer was grounded in bad source text.

If you omit `--debug-output`, inspection runs default to `inspect_outputs/hsr_v1_debug_pages.jsonl`, which is also gitignored.

```bash
python3 inspect_lore_hsr.py --query "Stellaron Hunters" --query-scope content --limit 3
```

Use `--query` when you know a lore phrase but not the page title. The default debugging-friendly scope is cleaned content; you can also search `title`, `raw`, or `any`.

`extract_lore_hsr.py` is now extraction-only. `inspect_lore_hsr.py` owns the page-level debugging workflow.
