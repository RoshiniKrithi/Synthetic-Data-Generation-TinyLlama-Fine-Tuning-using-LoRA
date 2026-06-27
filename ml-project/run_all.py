"""
run_all.py — One-click entry point for the full ML pipeline.

Phases:
  1. Scrape Wikipedia article
  2. Generate synthetic Q&A pairs with Mistral + Ollama
  3. Analyse the dataset and produce charts
  4. Fine-tune TinyLlama with LoRA
  5. Evaluate base vs fine-tuned model
  6. Prepare Inference pipelines
  7. Launch the Gradio web interface

Run:
  python run_all.py --help
  python run_all.py                          # full pipeline
  python run_all.py --phases 1 2 3           # selective phases
  python run_all.py --phases 7               # only launch UI
  python run_all.py --url https://en.wikipedia.org/wiki/BERT_(language_model)
  python run_all.py --resume                 # skip already-completed phases
  python run_all.py --reset-checkpoint       # clear saved checkpoint
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on the import path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.utils.helpers import setup_logging, load_config, ensure_dirs, format_time
from src.utils.checkpoint import PipelineCheckpoint
from src.utils.progress import PipelineDisplay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic Data Generation + TinyLlama LoRA Fine-Tuning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phases",
        nargs="*",
        type=int,
        choices=range(1, 8),
        metavar="N",
        help="Phases to run (1-7). Default: all phases.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Wikipedia URL to scrape (overrides config).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "config.yaml"),
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip Phase 4 (fine-tuning) — useful when adapter already exists.",
    )
    parser.add_argument(
        "--use-agents",
        action="store_true",
        help=(
            "Use the Multi-Agent pipeline for Phase 2 (Generator→Critic→Refiner). "
            "Produces higher-quality training data at the cost of more Ollama calls."
        ),
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip phases already recorded as done in the checkpoint file.",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Clear the checkpoint file before running (forces full re-run).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the rich progress display (plain log output only).",
    )
    return parser.parse_args()


def run_phase(
    phase: int,
    args: argparse.Namespace,
    config: dict,
    state: dict,
    display: PipelineDisplay,
    checkpoint: PipelineCheckpoint,
) -> None:
    """Dispatch execution to the correct phase module.

    Args:
        phase: Integer phase number (1-7).
        args: Parsed CLI arguments.
        config: Loaded project config.
        state: Shared state dict to pass data between phases.
        display: PipelineDisplay instance for status updates.
        checkpoint: PipelineCheckpoint instance for persisting progress.
    """
    logger = logging.getLogger("run_all")

    # ------------------------------------------------------------------
    # Resume logic — skip if already done
    # ------------------------------------------------------------------
    if args.resume and checkpoint.is_done(phase):
        logger.info("Phase %d already complete (checkpoint). Skipping.", phase)
        display.update(phase, status="skipped", detail="resumed from checkpoint")
        # Reload persisted metadata back into state if available
        meta = checkpoint.phase_metadata(phase)
        if phase == 1 and "num_chunks" in meta:
            logger.info("Checkpoint: Phase 1 produced %d chunks.", meta["num_chunks"])
        if phase == 2 and "num_pairs" in meta:
            logger.info("Checkpoint: Phase 2 produced %d Q&A pairs.", meta["num_pairs"])
        return

    t0 = time.perf_counter()
    sep = "=" * 60
    display.update(phase, status="running")

    # ------------------------------------------------------------------
    # Phase dispatch
    # ------------------------------------------------------------------
    if phase == 1:
        logger.info("%s\nPHASE 1 — Wikipedia Scraper\n%s", sep, sep)
        from src.scraper.wiki_scraper import WikipediaScraper
        scraper = WikipediaScraper(config)
        result = scraper.run(url=args.url)
        state["chunks"] = result["chunks"]
        elapsed = time.perf_counter() - t0
        detail = f"{len(state['chunks'])} chunks"
        logger.info("Phase 1 complete. Chunks: %d | Time: %s", len(state["chunks"]), format_time(elapsed))
        display.update(phase, status="done", detail=detail)
        checkpoint.mark_done(phase, metadata={"num_chunks": len(state["chunks"])})

    elif phase == 2:
        use_agents = getattr(args, "use_agents", False)
        mode_label = "Multi-Agent" if use_agents else "Classic"
        logger.info("%s\nPHASE 2 — Synthetic Q&A Generation (%s)\n%s", sep, mode_label, sep)

        if not state.get("chunks"):
            chunks_path = Path(config["paths"]["data_processed"]) / "chunks.json"
            if chunks_path.exists():
                import json
                state["chunks"] = json.loads(chunks_path.read_text(encoding="utf-8"))
                logger.info("Loaded %d chunks from disk.", len(state["chunks"]))
            else:
                logger.error("No chunks available. Run Phase 1 first.")
                display.update(phase, status="failed", detail="missing chunks")
                return

        if use_agents:
            display.update(phase, status="running", detail="Multi-Agent pipeline…")
            from src.generator.agents import MultiAgentOrchestrator
            orchestrator = MultiAgentOrchestrator(config)
            state["qa_pairs"] = orchestrator.run(state["chunks"])
        else:
            from src.generator.qa_generator import QAGenerator
            gen = QAGenerator(config)
            state["qa_pairs"] = gen.run(state["chunks"])

        elapsed = time.perf_counter() - t0
        detail = f"{len(state['qa_pairs'])} Q&A pairs [{mode_label}]"
        logger.info("Phase 2 complete. Q&A pairs: %d | Time: %s", len(state["qa_pairs"]), format_time(elapsed))
        display.update(phase, status="done", detail=detail)
        checkpoint.mark_done(phase, metadata={"num_pairs": len(state["qa_pairs"]), "mode": mode_label})

    elif phase == 3:
        logger.info("%s\nPHASE 3 — Dataset Analysis\n%s", sep, sep)
        from src.analysis.data_analysis import run_analysis
        stats = run_analysis(config)
        elapsed = time.perf_counter() - t0
        total = stats.get("total_pairs", 0)
        detail = f"{total} pairs analysed"
        logger.info("Phase 3 complete. Stats: %s | Time: %s", stats, format_time(elapsed))
        display.update(phase, status="done", detail=detail)
        checkpoint.mark_done(phase, metadata=stats)

    elif phase == 4:
        if args.skip_training:
            logger.info("PHASE 4 skipped (--skip-training flag).")
            display.update(phase, status="skipped", detail="--skip-training")
            return

        logger.info("%s\nPHASE 4 — LoRA Fine-Tuning\n%s", sep, sep)
        if not state.get("qa_pairs"):
            import json
            qa_path = Path(config["paths"]["data_synthetic"]) / "synthetic_qa.json"
            if qa_path.exists():
                state["qa_pairs"] = json.loads(qa_path.read_text(encoding="utf-8"))
                logger.info("Loaded %d Q&A pairs from disk.", len(state["qa_pairs"]))
            else:
                logger.error("No Q&A pairs available. Run Phase 2 first.")
                display.update(phase, status="failed", detail="missing qa pairs")
                return

        from src.training.fine_tune import LoRAFineTuner
        tuner = LoRAFineTuner(config)
        adapter_dir = tuner.run(state["qa_pairs"])
        state["adapter_dir"] = adapter_dir
        elapsed = time.perf_counter() - t0
        logger.info("Phase 4 complete. Adapter saved → %s | Time: %s", adapter_dir, format_time(elapsed))
        display.update(phase, status="done", detail="adapter saved")
        checkpoint.mark_done(phase, metadata={"adapter_dir": str(adapter_dir)})

    elif phase == 5:
        logger.info("%s\nPHASE 5 — Evaluation\n%s", sep, sep)
        if not state.get("qa_pairs"):
            import json
            qa_path = Path(config["paths"]["data_synthetic"]) / "synthetic_qa.json"
            if qa_path.exists():
                state["qa_pairs"] = json.loads(qa_path.read_text(encoding="utf-8"))

        from src.inference.pipeline import InferencePipeline
        from src.evaluation.evaluate import ModelEvaluator

        base_pipe = InferencePipeline(config, use_fine_tuned=False)
        base_pipe.load()
        ft_pipe = InferencePipeline(config, use_fine_tuned=True)
        ft_pipe.load()

        state["base_pipeline"] = base_pipe
        state["ft_pipeline"] = ft_pipe

        evaluator = ModelEvaluator(config)
        report = evaluator.run(state["qa_pairs"], base_pipe, ft_pipe)
        state["eval_report"] = report

        elapsed = time.perf_counter() - t0
        bleu = report.get("base_model", {}).get("metrics", {}).get("bleu", "?")
        detail = f"BLEU base={bleu}"
        logger.info("Phase 5 complete. Time: %s", format_time(elapsed))
        display.update(phase, status="done", detail=detail)
        checkpoint.mark_done(phase, metadata={"bleu_base": bleu})

    elif phase == 6:
        logger.info("%s\nPHASE 6 — Inference Pipeline Ready\n%s", sep, sep)
        from src.inference.pipeline import InferencePipeline
        if not state.get("base_pipeline"):
            state["base_pipeline"] = InferencePipeline(config, use_fine_tuned=False)
            state["base_pipeline"].load()
        if not state.get("ft_pipeline"):
            state["ft_pipeline"] = InferencePipeline(config, use_fine_tuned=True)
            state["ft_pipeline"].load()
        elapsed = time.perf_counter() - t0
        logger.info("Phase 6 complete. Pipelines ready. Time: %s", format_time(elapsed))
        display.update(phase, status="done", detail="pipelines loaded")
        checkpoint.mark_done(phase)

    elif phase == 7:
        logger.info("%s\nPHASE 7 — Gradio Web App\n%s", sep, sep)
        from src.inference.pipeline import InferencePipeline
        from src.ui.gradio_app import GradioApp

        if not state.get("base_pipeline"):
            state["base_pipeline"] = InferencePipeline(config, use_fine_tuned=False)
            state["base_pipeline"].load()
        if not state.get("ft_pipeline"):
            state["ft_pipeline"] = InferencePipeline(config, use_fine_tuned=True)
            state["ft_pipeline"].load()

        display.update(phase, status="running", detail="launching Gradio…")
        app = GradioApp(config, state["base_pipeline"], state["ft_pipeline"])
        app.launch(share=args.share)


def main() -> None:
    args = parse_args()

    # Change working directory to project root so relative paths work
    os.chdir(ROOT)

    config = load_config(str(ROOT / "configs" / "config.yaml"))
    ensure_dirs(config)
    setup_logging(config["logging"]["file"], config["logging"]["level"])
    logger = logging.getLogger("run_all")

    # ------------------------------------------------------------------
    # Checkpoint setup
    # ------------------------------------------------------------------
    checkpoint_path = ROOT / "logs" / "pipeline_checkpoint.json"
    checkpoint = PipelineCheckpoint(checkpoint_path)

    if args.reset_checkpoint:
        checkpoint.reset()
        logger.info("Checkpoint cleared.")

    if checkpoint.completed_phases():
        logger.info("Checkpoint state:\n%s", checkpoint.summary())

    # ------------------------------------------------------------------
    # Progress display
    # ------------------------------------------------------------------
    display = PipelineDisplay()
    if not args.no_progress:
        display.start()

    # ------------------------------------------------------------------
    # Phase selection
    # ------------------------------------------------------------------
    phases = args.phases if args.phases else list(range(1, 8))
    logger.info("Running phases: %s", phases)

    state: dict = {}
    total_start = time.perf_counter()
    failed_phases: list[int] = []

    for phase in sorted(phases):
        try:
            run_phase(phase, args, config, state, display, checkpoint)
        except KeyboardInterrupt:
            logger.info("Interrupted at phase %d.", phase)
            display.update(phase, status="failed", detail="interrupted")
            display.stop()
            sys.exit(0)
        except Exception as exc:
            logger.exception("Phase %d failed: %s", phase, exc)
            display.update(phase, status="failed", detail=str(exc)[:60])
            failed_phases.append(phase)
            logger.info("Continuing with next phase …")

    if not args.no_progress:
        display.stop()

    total_elapsed = format_time(time.perf_counter() - total_start)
    if failed_phases:
        logger.warning(
            "Pipeline finished with failures in phases %s. Total time: %s",
            failed_phases,
            total_elapsed,
        )
    else:
        logger.info("All phases complete. Total time: %s", total_elapsed)


if __name__ == "__main__":
    main()
