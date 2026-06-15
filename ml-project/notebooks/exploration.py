"""
Interactive exploration script — run cell by cell in a REPL or Jupyter.
Demonstrates each pipeline stage with small examples.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.helpers import load_config, setup_logging

config = load_config("configs/config.yaml")
setup_logging()

# ── Cell 1: Scrape ─────────────────────────────────────────────────────
from src.scraper.wiki_scraper import WikipediaScraper
scraper = WikipediaScraper(config)
result = scraper.run()
print(f"Chunks: {len(result['chunks'])}")
print("First chunk preview:", result['chunks'][0]['text'][:200])

# ── Cell 2: Generate Q&A (needs Ollama running) ────────────────────────
from src.generator.qa_generator import QAGenerator
gen = QAGenerator(config)
pairs = gen.run(result['chunks'][:3])  # Quick test on 3 chunks
print(f"Generated {len(pairs)} Q&A pairs")
if pairs:
    print("Sample:", pairs[0])

# ── Cell 3: Analysis ───────────────────────────────────────────────────
from src.analysis.data_analysis import run_analysis
stats = run_analysis(config)
print("Dataset stats:", stats)

# ── Cell 4: Quick inference test (no fine-tuning needed) ───────────────
from src.inference.pipeline import InferencePipeline
pipe = InferencePipeline(config, use_fine_tuned=False)
pipe.load()
resp = pipe.generate("What is the transformer architecture?")
print("Base model response:", resp['response'][:300])
print(f"Latency: {resp['latency_s']}s | Tokens: {resp['token_count']}")
