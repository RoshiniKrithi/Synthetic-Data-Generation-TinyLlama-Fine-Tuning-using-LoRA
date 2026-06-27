"""
src/utils/checkpoint.py — Pipeline checkpoint & resume system.

Persists phase-level completion state to a JSON file so that a crashed or
interrupted pipeline can resume from the last successfully completed phase
rather than restarting from scratch.

Usage::

    from src.utils.checkpoint import PipelineCheckpoint

    cp = PipelineCheckpoint("logs/pipeline_checkpoint.json")
    if not cp.is_done(1):
        # ... run phase 1 ...
        cp.mark_done(1, metadata={"chunks": 30})

    # Resume from last checkpoint
    completed = cp.completed_phases()
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PipelineCheckpoint:
    """Lightweight JSON-backed checkpoint store for pipeline phases.

    Args:
        path: Path to the JSON checkpoint file.  Parent directories are
              created automatically.
    """

    def __init__(self, path: str | Path = "logs/pipeline_checkpoint.json") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mark_done(self, phase: int, metadata: dict | None = None) -> None:
        """Record that *phase* completed successfully.

        Args:
            phase: Integer phase number (1–7).
            metadata: Optional dict of key statistics to persist alongside
                      the completion record (e.g. chunk count, pair count).
        """
        key = str(phase)
        self._data[key] = {
            "completed": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "metadata": metadata or {},
        }
        self._save()
        logger.debug("Checkpoint: phase %d marked as done.", phase)

    def is_done(self, phase: int) -> bool:
        """Return True if *phase* has been recorded as completed.

        Args:
            phase: Integer phase number.

        Returns:
            True if the phase completed in a previous run.
        """
        return self._data.get(str(phase), {}).get("completed", False)

    def completed_phases(self) -> list[int]:
        """Return a sorted list of all completed phase numbers.

        Returns:
            List of ints, e.g. ``[1, 2, 3]``.
        """
        return sorted(
            int(k) for k, v in self._data.items() if v.get("completed", False)
        )

    def phase_metadata(self, phase: int) -> dict:
        """Return the metadata dict stored for *phase*.

        Args:
            phase: Integer phase number.

        Returns:
            Metadata dict (may be empty if phase not recorded or no metadata).
        """
        return self._data.get(str(phase), {}).get("metadata", {})

    def reset(self, phases: list[int] | None = None) -> None:
        """Remove checkpoint records.

        Args:
            phases: Specific phase numbers to reset.  Pass ``None`` to clear
                    all phases (full reset).
        """
        if phases is None:
            self._data.clear()
            logger.info("Checkpoint: all phases reset.")
        else:
            for p in phases:
                self._data.pop(str(p), None)
            logger.info("Checkpoint: phases %s reset.", phases)
        self._save()

    def summary(self) -> str:
        """Return a human-readable summary of the checkpoint state.

        Returns:
            Multi-line string describing which phases are done.
        """
        if not self._data:
            return "No phases completed yet."
        lines = ["Pipeline Checkpoint Summary", "=" * 40]
        for phase in sorted(int(k) for k in self._data):
            entry = self._data[str(phase)]
            ts = entry.get("timestamp", "unknown")
            meta = entry.get("metadata", {})
            meta_str = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else "—"
            lines.append(f"  Phase {phase}: ✓  ({ts})  [{meta_str}]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        """Load existing checkpoint file or return empty dict."""
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                logger.debug("Checkpoint loaded from %s", self._path)
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load checkpoint (%s) — starting fresh.", exc)
        return {}

    def _save(self) -> None:
        """Persist current state to the JSON file (atomic-ish write)."""
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(self._path)
