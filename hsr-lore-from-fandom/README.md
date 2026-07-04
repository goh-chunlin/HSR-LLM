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

## Smaller Debug Workflows

The XML wiki dump is large enough that it should not be your debugging unit. Use the extractor to create compact JSONL artifacts instead.

```bash
python3 extract_lore_hsr.py --limit 20
```

That writes only the first 20 cleaned lore pages so you can iterate without regenerating the full dataset.

If you omit `--output`, limited extraction runs default to `inspect_outputs/hsr_v1_raw_lore_sample.jsonl`, which is gitignored for tester-friendly scratch output.

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
