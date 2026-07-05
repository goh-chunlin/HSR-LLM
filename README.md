# Honkai: Star Rail — Lore LoRA

Two parallel tracks for building a Herta-persona language model.

## Track A — Lore RAG Space (`hsr-lore-from-fandom/`)

A Gradio web app deployed to Hugging Face Spaces. Retrieves HSR lore using a
hybrid BM25 + FAISS pipeline and synthesises answers with Llama 3.1 8B Instruct.

**Runtime files** (built on CI/CD pipeline `.github/workflows/refresh_lore_and_sync_space.yml`):
- `hsr_v1_chunks.json` — text chunks from the HSR Fandom wiki
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

## GitHub + Hugging Face Deployment Playbook

This section is the long-term checklist for managing one monorepo while
deploying only `hsr-lore-from-fandom/` to Hugging Face Space.

### Repository model

1. `origin` (GitHub): stores the full repository.
2. `space` (Hugging Face): stores only the deployment view generated from
	`hsr-lore-from-fandom/`.
3. Deployment method: `git subtree split --prefix hsr-lore-from-fandom`.

### One-time setup (new machine)

1. Clone your GitHub repository.

```bash
git clone <your-github-repo-url>
cd honkai-star-rail-lore-lora
```

2. Install Git LFS.

```bash
git lfs install
```

3. Verify remotes.

```bash
git remote -v
```

Expected:

1. `origin` points to GitHub.
2. `space` points to `https://huggingface.co/spaces/goh-chunlin/hsr-lore`.

If `space` is missing:

```bash
git remote add space https://huggingface.co/spaces/goh-chunlin/hsr-lore
```

### First deployment to Hugging Face Space

1. Commit everything normally to your monorepo.

```bash
git add .
git commit -m "chore: prepare initial Space deployment"
git push origin main
```

2. Upload LFS objects to HF remote.

```bash
git lfs push --all space
```

3. Push only the Space subfolder to HF `main`.

```bash
git push space $(git subtree split --prefix hsr-lore-from-fandom):main --force
```

Notes:

1. You will be prompted for credentials; use your HF access token as password.
2. `--force` is required when Space history diverges from your local split.

### Daily update workflow

1. Make code changes in the monorepo.
2. Commit and push to GitHub.
3. Deploy subfolder snapshot to Space.

```bash
git add .
git commit -m "feat: update retrieval logic"
git push origin main
git lfs push --all space
git push space $(git subtree split --prefix hsr-lore-from-fandom):main --force
```

### Space secrets (required for LLM calls)

In Hugging Face Space settings:

1. Go to `Settings`.
2. Open `Variables and secrets`.
3. Add secret `HF_TOKEN` with a token that has inference access.

Without this secret, the app will fail with API key/authentication errors.

### Pre-deploy checklist

1. `hsr-lore-from-fandom/.gitattributes` exists and includes LFS rules for
	`*.faiss` and `hsr_v1_chunks.json`.
2. Runtime files exist in Space folder:
	`hsr-lore-from-fandom/my_hsr_1.0_index.faiss` and
	`hsr-lore-from-fandom/hsr_v1_chunks.json`.
3. `hsr-lore-from-fandom/README.md` contains Space YAML frontmatter.
4. `hsr-lore-from-fandom/requirements.txt` includes runtime dependencies.

### Troubleshooting guide

1. Push rejected (non-fast-forward):

```bash
git push space $(git subtree split --prefix hsr-lore-from-fandom):main --force
```

2. FAISS error `Index type 0x73726576 ("vers") not recognized`:
	This means FAISS is reading a Git LFS pointer text file, not real binary data.
	Fix by ensuring `.gitattributes` is present inside `hsr-lore-from-fandom/`,
	then run:

```bash
git lfs push --all space
git push space $(git subtree split --prefix hsr-lore-from-fandom):main --force
```

3. Inference error `You must provide an api_key`:
	Add `HF_TOKEN` in Space secrets and restart/rebuild the Space.

4. Space builds but app behaves outdated:

```bash
git add .
git commit -m "chore: redeploy latest Space state"
git push origin main
git push space $(git subtree split --prefix hsr-lore-from-fandom):main --force
```

### Useful commands quick reference

```bash
# show remotes
git remote -v

# show recent commits
git log --oneline -n 10

# verify LFS tracked files
git lfs ls-files

# deploy Space subfolder
git push space $(git subtree split --prefix hsr-lore-from-fandom):main --force
```
