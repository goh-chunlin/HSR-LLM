# Honkai: Star Rail — Lore LoRA

Two parallel tracks for building a Herta-persona language model.

## Track A — Lore RAG Space (`hsr-lore-from-fandom/`)

A Gradio web app deployed to Hugging Face Spaces. Retrieves HSR lore using a
hybrid BM25 + FAISS pipeline and synthesises answers with Llama 3.1 8B Instruct.

**Runtime files** (tracked via Git LFS):
- `hsr_v1_chunks.json` — 22 k text chunks from the HSR Fandom wiki
- `my_hsr_1.0_index.faiss` — pre-built dense vector index

**Build-time scripts** (run locally, outputs committed via LFS):
- `extract_lore_hsr.py` — parses the Fandom MediaWiki XML dump
- `build_lore_vector_db.py` — chunks text and builds the FAISS index

## Track B — Herta Persona LoRA (`data/`, `adapters/`)

Fine-tuning dataset and adapter checkpoints for teaching a base model to
speak like Herta and reject off-topic questions.

- `data/train.jsonl` / `data/valid.jsonl` — curated dialogue pairs
- `adapters/` — MLX LoRA checkpoint snapshots

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r hsr-lore-from-fandom/requirements.txt
```
