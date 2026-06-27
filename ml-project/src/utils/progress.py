"""
src/utils/progress.py — Rich terminal progress display.

Provides a RichProgress context manager that wraps the ``rich`` library to
produce beautiful, informative progress bars and a live pipeline status table
in the terminal.  Falls back gracefully to plain ``tqdm`` if ``rich`` is not
installed.

Usage::

    from src.utils.progress import PipelineDisplay

    PHASES = {
        1: "Wikipedia Scraper",
        2: "Q&A Generation",
        3: "Dataset Analysis",
        4: "LoRA Fine-Tuning",
        5: "Model Evaluation",
        6: "Inference Setup",
        7: "Gradio Web App",
    }

    display = PipelineDisplay(phases=PHASES)
    display.start()
    display.update(1, status="running")
    # ... do work ...
    display.update(1, status="done", detail="30 chunks")
    display.stop()
"""
from __future__ import annotations

import time
from typing import Optional


# ---------------------------------------------------------------------------
# Pipeline status display
# ---------------------------------------------------------------------------

PHASE_NAMES = {
    1: "Wikipedia Scraper",
    2: "Synthetic Q&A Generation",
    3: "Dataset Analysis",
    4: "LoRA Fine-Tuning",
    5: "Model Evaluation",
    6: "Inference Pipeline",
    7: "Gradio Web App",
}

STATUS_ICONS = {
    "pending":  "⏸",
    "running":  "⚙",
    "done":     "✅",
    "skipped":  "⏭",
    "failed":   "❌",
}


class PipelineDisplay:
    """Terminal display for the ML pipeline using ``rich`` if available.

    Args:
        phases: Mapping of phase number → display name.  Defaults to the
                built-in ``PHASE_NAMES`` dict.
        show_spinner: Whether to show a spinner next to running phases.
    """

    def __init__(
        self,
        phases: dict[int, str] | None = None,
        show_spinner: bool = True,
    ) -> None:
        self._phases = phases or PHASE_NAMES
        self._show_spinner = show_spinner
        self._statuses: dict[int, dict] = {
            p: {"status": "pending", "detail": "", "elapsed": 0.0}
            for p in self._phases
        }
        self._start_times: dict[int, float] = {}
        self._rich_available = self._check_rich()
        self._live = None
        self._console = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the display.  Call once before any ``update()`` calls."""
        if self._rich_available:
            self._start_rich()
        else:
            self._print_header()

    def update(
        self,
        phase: int,
        status: str = "running",
        detail: str = "",
    ) -> None:
        """Update the status of a phase.

        Args:
            phase: Phase number.
            status: One of ``pending``, ``running``, ``done``, ``skipped``,
                    ``failed``.
            detail: Short detail string (e.g. "30 chunks produced").
        """
        if status == "running" and phase not in self._start_times:
            self._start_times[phase] = time.perf_counter()

        elapsed = 0.0
        if phase in self._start_times:
            elapsed = time.perf_counter() - self._start_times[phase]

        self._statuses[phase] = {
            "status": status,
            "detail": detail,
            "elapsed": elapsed,
        }

        if self._rich_available and self._live:
            self._live.update(self._build_table())
        else:
            icon = STATUS_ICONS.get(status, "?")
            name = self._phases.get(phase, f"Phase {phase}")
            detail_str = f"  ({detail})" if detail else ""
            elapsed_str = f"  [{elapsed:.1f}s]" if elapsed else ""
            print(f"  {icon}  Phase {phase}: {name}{detail_str}{elapsed_str}")

    def stop(self) -> None:
        """Stop the live display and print a final summary."""
        if self._rich_available and self._live:
            self._live.update(self._build_table())
            self._live.stop()
        else:
            self._print_summary()

    # ------------------------------------------------------------------
    # Rich helpers
    # ------------------------------------------------------------------

    def _check_rich(self) -> bool:
        try:
            import rich  # noqa: F401
            return True
        except ImportError:
            return False

    def _start_rich(self) -> None:
        from rich.console import Console
        from rich.live import Live

        self._console = Console()
        self._live = Live(
            self._build_table(),
            console=self._console,
            refresh_per_second=4,
        )
        self._live.start()

    def _build_table(self):
        from rich.table import Table
        from rich import box
        from rich.text import Text

        table = Table(
            title="🤖 TinyLlama Fine-Tuning Pipeline",
            box=box.ROUNDED,
            show_footer=False,
            expand=True,
        )
        table.add_column("Phase", style="bold cyan", justify="center", width=7)
        table.add_column("Name", style="white", min_width=28)
        table.add_column("Status", justify="center", width=10)
        table.add_column("Detail", style="dim", min_width=20)
        table.add_column("Elapsed", justify="right", width=10)

        STATUS_STYLES = {
            "pending": "dim",
            "running": "yellow bold",
            "done":    "green bold",
            "skipped": "blue",
            "failed":  "red bold",
        }

        for p in sorted(self._phases):
            info = self._statuses[p]
            status = info["status"]
            icon = STATUS_ICONS.get(status, "?")
            style = STATUS_STYLES.get(status, "")
            elapsed = info["elapsed"]
            elapsed_str = f"{elapsed:.1f}s" if elapsed > 0 else "—"

            table.add_row(
                str(p),
                self._phases[p],
                Text(f"{icon} {status}", style=style),
                info["detail"] or "—",
                elapsed_str,
            )

        return table

    # ------------------------------------------------------------------
    # Plain-text fallback helpers
    # ------------------------------------------------------------------

    def _print_header(self) -> None:
        print("\n" + "=" * 60)
        print("  🤖  TinyLlama Fine-Tuning Pipeline")
        print("=" * 60)

    def _print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("  Pipeline Summary")
        print("=" * 60)
        for p in sorted(self._phases):
            info = self._statuses[p]
            icon = STATUS_ICONS.get(info["status"], "?")
            name = self._phases[p]
            detail = f"  ({info['detail']})" if info["detail"] else ""
            elapsed = f"  [{info['elapsed']:.1f}s]" if info["elapsed"] else ""
            print(f"  {icon}  Phase {p}: {name}{detail}{elapsed}")
        print("=" * 60 + "\n")
