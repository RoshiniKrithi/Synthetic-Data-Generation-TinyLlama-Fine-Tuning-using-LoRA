"""
quick_generate.py — Run only Phases 1-3 (scrape + generate + analyse).

Much faster than the full pipeline — useful for:
  - Testing the Ollama connection
  - Building the dataset before fine-tuning
  - Demos where you already have the dataset

Usage:
  python quick_generate.py
  python quick_generate.py --url https://en.wikipedia.org/wiki/BERT_(language_model)
  python quick_generate.py --chunks 10 --pairs 50
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.utils.helpers import load_config, setup_logging, ensure_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick data generation pipeline (Phases 1-3)")
    p.add_argument("--url", default=None, help="Wikipedia URL to scrape")
    p.add_argument("--chunks", type=int, default=None, help="Override max chunks from config")
    p.add_argument("--pairs", type=int, default=None, help="Override max Q&A pairs from config")
    p.add_argument(
        "--use-agents",
        action="store_true",
        help="Use the Multi-Agent pipeline (Generator→Critic→Refiner) for Phase 2.",
    )
    p.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    setup_logging(config["logging"]["file"])
    logger = logging.getLogger("quick_generate")

    if args.chunks:
        config["scraper"]["max_chunks"] = args.chunks
    if args.pairs:
        config["generator"]["max_qa_pairs"] = args.pairs
        if "agents" in config:
            config["agents"]["max_qa_pairs"] = args.pairs

    # Phase 1 — Scrape
    logger.info("─── Phase 1: Scraping Wikipedia ───")
    from src.scraper.wiki_scraper import WikipediaScraper
    scraper = WikipediaScraper(config)
    scrape_result = scraper.run(url=args.url)
    chunks = scrape_result["chunks"]
    logger.info("[OK] %d chunks scraped.", len(chunks))

    # Phase 2 — Generate
    logger.info("─── Phase 2: Generating Q&A pairs ───")
    if args.use_agents:
        logger.info("Using Multi-Agent generation (Generator → Critic → Refiner).")
        from src.generator.agents import MultiAgentOrchestrator
        orchestrator = MultiAgentOrchestrator(config)
        pairs = orchestrator.run(chunks)
    else:
        logger.info("Using Classic generation.")
        logger.info("Make sure Ollama is running: ollama serve && ollama pull mistral")
        from src.generator.qa_generator import QAGenerator
        gen = QAGenerator(config)
        pairs = gen.run(chunks)
    logger.info("[OK] %d Q&A pairs generated.", len(pairs))

    # Phase 3 — Analyse
    logger.info("─── Phase 3: Dataset analysis ───")
    from src.analysis.data_analysis import run_analysis
    stats = run_analysis(config)
    logger.info("[OK] Dataset stats: %s", stats)

    print("\n" + "=" * 50)
    print(f"Done! Generated {len(pairs)} Q&A pairs.")
    print(f"  JSON: {config['paths']['data_synthetic']}/synthetic_qa.json")
    print(f"  CSV:  {config['paths']['data_synthetic']}/synthetic_qa.csv")
    print(f"  Charts: {config['paths']['charts']}/")
    print("=" * 50)
    print("\nNext steps:")
    print("  Fine-tune: python run_all.py --phases 4")
    print("  Evaluate:  python run_all.py --phases 5")
    print("  Launch UI: python launch_ui.py --base-only")


if __name__ == "__main__":
    main()
