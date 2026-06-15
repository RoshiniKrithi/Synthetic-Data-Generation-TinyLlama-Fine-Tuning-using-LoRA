# Synthetic Data Gen + TinyLlama Fine-Tuning with LoRA

End-to-end Generative AI portfolio project: scrape Wikipedia → generate synthetic Q&A with Mistral → fine-tune TinyLlama-1.1B with LoRA → evaluate → Gradio web UI comparison.

## Run & Operate

### Gradio Web App (main workflow)
The **Gradio App** workflow runs automatically. It opens the UI at port 8000 immediately; TinyLlama model weights load in the background (takes 1–3 min on first run — they are cached after that).

### Full Pipeline (all phases)
```bash
cd ml-project
python run_all.py               # all 7 phases
python run_all.py --phases 1 2  # scrape + generate only
python run_all.py --phases 4    # fine-tune only
python run_all.py --url https://en.wikipedia.org/wiki/BERT_(language_model)
```

### Data Generation only (requires Ollama)
```bash
# Terminal 1: start Ollama
ollama serve
ollama pull mistral

# Terminal 2: generate data
cd ml-project && python quick_generate.py
```

### Launch UI only
```bash
cd ml-project
python launch_ui.py             # base + fine-tuned
python launch_ui.py --base-only # base model only (no adapter needed)
```

### TypeScript API Server (Node.js)
```bash
pnpm --filter @workspace/api-server run dev  # port 5000
pnpm run typecheck                            # full TS check
pnpm run build                                # build all packages
pnpm --filter @workspace/api-spec run codegen # regenerate API hooks
pnpm --filter @workspace/db run push          # push DB schema (dev only)
```

## Stack

### Python ML Pipeline
- Python 3.11
- PyTorch · HuggingFace Transformers · PEFT · Datasets · Accelerate
- Gradio 4.x (web UI)
- Ollama + Mistral (synthetic data generation)
- Matplotlib · Plotly (charts)
- BeautifulSoup4 (scraping)
- BLEU · ROUGE-L · BERTScore · Exact Match (evaluation)

### TypeScript API Server (existing workspace)
- pnpm workspaces, Node.js 24, TypeScript 5.9
- Express 5, PostgreSQL + Drizzle ORM, Zod, Orval codegen

## Where Things Live

```
ml-project/
├── configs/config.yaml       ← ALL hyperparameters + paths
├── src/
│   ├── scraper/              ← Phase 1: Wikipedia scraper
│   ├── generator/            ← Phase 2: Mistral Q&A generator
│   ├── analysis/             ← Phase 3: Dataset stats + charts
│   ├── training/             ← Phase 4: LoRA fine-tuning
│   ├── evaluation/           ← Phase 5: BLEU/ROUGE/BERTScore
│   ├── inference/            ← Phase 6: Inference pipeline
│   ├── ui/                   ← Phase 7: Gradio app
│   └── utils/                ← Shared helpers + logging
├── data/synthetic/           ← synthetic_qa.json + .csv
├── models/adapter/           ← LoRA adapter weights (after training)
├── reports/charts/           ← PNG comparison charts
├── run_all.py                ← One-command full pipeline
├── launch_ui.py              ← Gradio UI launcher
├── quick_generate.py         ← Phases 1-3 only
├── interview_prep.md         ← 20 interview Q&A with answers
└── resume_content.md         ← ATS resume bullets + LinkedIn copy

artifacts/api-server/         ← TypeScript Express API
lib/                          ← Shared TS libraries
```

## Architecture

```
Wikipedia URL → Scraper → Chunks → Mistral (Ollama) → synthetic_qa.json
→ TinyLlama + LoRA (PEFT) → adapter weights → Evaluation → Gradio UI
```

## User Preferences

- CPU-compatible pipeline (no GPU required)
- One-command execution
- Clean modular architecture
- Production-ready code quality (type hints, docstrings, logging, retry logic)

## Gotchas

- **Ollama must be running** before Phases 1-2: `ollama serve && ollama pull mistral`
- First model load downloads ~2 GB of weights — cached in `~/.cache/huggingface/` after that
- Fine-tuned model comparison only works after running Phase 4 (training); UI falls back to base model if adapter is missing
- Port 8000 is used by the Gradio workflow; TypeScript API server uses 8080 internally (proxied at /api)
- Run `pnpm --filter @workspace/api-spec run codegen` after any OpenAPI spec change
- `pnpm --filter @workspace/db run push` requires `DATABASE_URL` env var set

## Pointers

- See `ml-project/interview_prep.md` for 20 ML/GenAI interview questions
- See `ml-project/resume_content.md` for ATS resume bullets and LinkedIn description
- See `ml-project/configs/config.yaml` for all tunable hyperparameters
- See the `pnpm-workspace` skill for TS workspace structure details
