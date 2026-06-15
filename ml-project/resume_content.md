# Resume & LinkedIn Content — ML / GenAI Internship

## ATS-Optimised Resume Bullet Points

### Project Title
**Synthetic Data Generation + TinyLlama Fine-Tuning using LoRA** | Python · HuggingFace · PEFT · Gradio | 2024

### Impact-First Bullets (pick 4-5 for your resume)

- Built an end-to-end Generative AI pipeline that auto-generates 200-500 synthetic Q&A pairs from any Wikipedia article using Mistral (via Ollama) and fine-tunes TinyLlama-1.1B with LoRA, achieving measurable gains across BLEU, ROUGE-L, and BERTScore metrics.

- Implemented Parameter-Efficient Fine-Tuning (PEFT) with LoRA (r=16, α=32) on TinyLlama-1.1B, reducing trainable parameters by >99% vs full fine-tuning while preserving model quality — enabling training on CPU with <4 GB RAM.

- Engineered a multi-type prompt strategy (factual, reasoning, conceptual, definition) to generate diverse synthetic training data; applied MD5 deduplication and length filtering to maintain dataset quality across 200-500 Q&A pairs.

- Designed a modular 8-phase ML pipeline (scrape → generate → analyse → train → evaluate → infer → serve) with full logging, YAML config management, retry logic, and one-command execution via `run_all.py`.

- Deployed a professional Gradio web application providing real-time, side-by-side comparison of Base vs Fine-Tuned TinyLlama with temperature control, token count display, and result download — accessible via browser with zero setup.

- Evaluated model performance using four complementary NLP metrics (Exact Match, BLEU, ROUGE-L, BERTScore-F1), generating automated HTML/Markdown/JSON reports and matplotlib comparison charts for portfolio presentation.

---

## LinkedIn Project Description

**Project: Synthetic Data Generation + TinyLlama Fine-Tuning with LoRA**

I built a complete, production-ready Generative AI portfolio project that demonstrates the full modern LLM development workflow — from raw data collection to deployment.

**What it does:**
🔹 Scrapes any Wikipedia article with intelligent cleaning and chunking
🔹 Uses Mistral (Ollama) to generate 200–500 diverse synthetic Q&A pairs
🔹 Fine-tunes TinyLlama-1.1B using LoRA via HuggingFace PEFT — CPU compatible
🔹 Evaluates Base vs Fine-Tuned model with BLEU, ROUGE-L, and BERTScore
🔹 Serves a Gradio web app for real-time side-by-side model comparison

**Tech stack:** Python · PyTorch · HuggingFace Transformers · PEFT · Datasets · Gradio · Ollama · Mistral · Matplotlib · BeautifulSoup · Pandas

**Key learning:** This project taught me how industry teams build synthetic training pipelines to bootstrap fine-tuned models when labelled data is scarce — a critical skill in modern GenAI product development.

🔗 GitHub: [your-repo-link]

---

## GitHub Project Description (README tagline)

> End-to-end GenAI pipeline: Wikipedia scraping → Mistral synthetic Q&A generation → TinyLlama LoRA fine-tuning → automated evaluation → Gradio web app. CPU-compatible. One-command execution.

---

## Project Impact Statement

This project demonstrates proficiency across the **complete modern ML engineering lifecycle**:

| Domain | Skills Demonstrated |
|---|---|
| Data Engineering | Web scraping, text cleaning, chunking, deduplication |
| Generative AI | Prompt engineering, synthetic data generation, Ollama |
| ML Training | PEFT/LoRA fine-tuning, HuggingFace Trainer, gradient accumulation |
| MLOps | Config management, logging, checkpointing, experiment reproducibility |
| Evaluation | Multi-metric NLP evaluation, automated report generation |
| Deployment | Gradio web app, CPU/GPU compatibility, environment variables |

---

## Quantified Achievements for Interviews

Use these framings when asked about impact:

1. **"Reduced fine-tuning cost by >99%"** — LoRA trains <1% of parameters vs full fine-tuning
2. **"Generated 300+ training examples in minutes"** — vs weeks of manual annotation
3. **"Zero cloud spend"** — entire pipeline runs locally (CPU-compatible)
4. **"4 NLP metrics tracked automatically"** — EM, BLEU, ROUGE-L, BERTScore with report generation
5. **"8 modular phases"** — each independently runnable and testable
6. **"One-command execution"** — `python run_all.py` runs the entire pipeline end-to-end
