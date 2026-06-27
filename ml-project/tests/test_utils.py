"""
tests/test_utils.py — Unit tests for shared utility helpers and the
checkpoint/progress modules.

Tests cover:
  - format_time() human-readable duration formatting
  - save_json() / load_json() round-trip
  - load_config() YAML parsing
  - ensure_dirs() directory creation
  - PipelineCheckpoint: mark_done, is_done, completed_phases, reset, summary
  - PipelineDisplay: instantiation and update without crashing
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import format_time, save_json, load_json, ensure_dirs, load_config
from src.utils.checkpoint import PipelineCheckpoint


# ---------------------------------------------------------------------------
# format_time
# ---------------------------------------------------------------------------

class TestFormatTime:
    def test_seconds_only(self):
        assert format_time(45.9) == "45s"

    def test_minutes_and_seconds(self):
        assert format_time(125.0) == "2m 5s"

    def test_hours_minutes_seconds(self):
        result = format_time(3661.0)
        assert "h" in result
        assert "m" in result
        assert "s" in result

    def test_zero_seconds(self):
        assert format_time(0) == "0s"

    def test_exactly_one_minute(self):
        assert format_time(60.0) == "1m 0s"

    def test_exactly_one_hour(self):
        result = format_time(3600.0)
        assert result.startswith("1h")


# ---------------------------------------------------------------------------
# save_json / load_json
# ---------------------------------------------------------------------------

class TestJsonHelpers:
    def test_save_and_load_roundtrip(self, tmp_path):
        data = {"key": "value", "numbers": [1, 2, 3], "nested": {"a": True}}
        path = str(tmp_path / "test.json")
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_save_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "file.json")
        save_json({"x": 1}, path)
        assert Path(path).exists()

    def test_save_uses_indent(self, tmp_path):
        path = str(tmp_path / "formatted.json")
        save_json({"a": 1}, path, indent=4)
        content = Path(path).read_text()
        assert "    " in content  # 4-space indent

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            load_json(str(tmp_path / "nonexistent.json"))


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_yaml(self, tmp_path):
        config_data = {"key": "value", "number": 42, "nested": {"x": True}}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config_data), encoding="utf-8")
        loaded = load_config(str(config_path))
        assert loaded["key"] == "value"
        assert loaded["number"] == 42
        assert loaded["nested"]["x"] is True

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            load_config(str(tmp_path / "missing.yaml"))


# ---------------------------------------------------------------------------
# ensure_dirs
# ---------------------------------------------------------------------------

class TestEnsureDirs:
    def test_creates_all_paths(self, tmp_path):
        config = {
            "paths": {
                "data_raw": str(tmp_path / "raw"),
                "data_processed": str(tmp_path / "proc"),
                "logs": str(tmp_path / "logs"),
            }
        }
        ensure_dirs(config)
        for key, path in config["paths"].items():
            assert Path(path).is_dir(), f"{key} should be a directory"

    def test_does_not_crash_if_already_exists(self, tmp_path):
        config = {"paths": {"existing": str(tmp_path)}}
        ensure_dirs(config)  # Should not raise


# ---------------------------------------------------------------------------
# PipelineCheckpoint
# ---------------------------------------------------------------------------

class TestPipelineCheckpoint:
    def test_initially_not_done(self, tmp_path):
        cp = PipelineCheckpoint(tmp_path / "cp.json")
        assert not cp.is_done(1)
        assert not cp.is_done(7)

    def test_mark_done_sets_flag(self, tmp_path):
        cp = PipelineCheckpoint(tmp_path / "cp.json")
        cp.mark_done(1)
        assert cp.is_done(1)

    def test_mark_done_with_metadata(self, tmp_path):
        cp = PipelineCheckpoint(tmp_path / "cp.json")
        cp.mark_done(2, metadata={"num_pairs": 300})
        assert cp.phase_metadata(2)["num_pairs"] == 300

    def test_completed_phases_returns_sorted_list(self, tmp_path):
        cp = PipelineCheckpoint(tmp_path / "cp.json")
        cp.mark_done(3)
        cp.mark_done(1)
        cp.mark_done(2)
        assert cp.completed_phases() == [1, 2, 3]

    def test_persists_to_disk(self, tmp_path):
        path = tmp_path / "cp.json"
        cp1 = PipelineCheckpoint(path)
        cp1.mark_done(5, metadata={"bleu": 18.5})

        # Load a new instance from the same file
        cp2 = PipelineCheckpoint(path)
        assert cp2.is_done(5)
        assert cp2.phase_metadata(5)["bleu"] == 18.5

    def test_reset_all_phases(self, tmp_path):
        cp = PipelineCheckpoint(tmp_path / "cp.json")
        cp.mark_done(1)
        cp.mark_done(2)
        cp.reset()
        assert cp.completed_phases() == []

    def test_reset_specific_phases(self, tmp_path):
        cp = PipelineCheckpoint(tmp_path / "cp.json")
        cp.mark_done(1)
        cp.mark_done(2)
        cp.mark_done(3)
        cp.reset(phases=[2])
        assert cp.is_done(1)
        assert not cp.is_done(2)
        assert cp.is_done(3)

    def test_summary_includes_phase_numbers(self, tmp_path):
        cp = PipelineCheckpoint(tmp_path / "cp.json")
        cp.mark_done(1)
        cp.mark_done(4)
        summary = cp.summary()
        assert "Phase 1" in summary
        assert "Phase 4" in summary

    def test_empty_summary_message(self, tmp_path):
        cp = PipelineCheckpoint(tmp_path / "cp.json")
        assert "No phases" in cp.summary()

    def test_handles_corrupted_file_gracefully(self, tmp_path):
        path = tmp_path / "cp.json"
        path.write_text("not valid json", encoding="utf-8")
        cp = PipelineCheckpoint(path)  # Should not raise
        assert cp.completed_phases() == []

    def test_mark_done_records_timestamp(self, tmp_path):
        cp = PipelineCheckpoint(tmp_path / "cp.json")
        cp.mark_done(1)
        raw = json.loads((tmp_path / "cp.json").read_text())
        assert "timestamp" in raw["1"]


# ---------------------------------------------------------------------------
# PipelineDisplay — smoke tests (no real terminal needed)
# ---------------------------------------------------------------------------

class TestPipelineDisplay:
    def test_instantiation_without_rich(self):
        """Should not raise even when rich is unavailable."""
        from src.utils.progress import PipelineDisplay
        display = PipelineDisplay()
        assert display is not None

    def test_update_does_not_raise(self):
        from src.utils.progress import PipelineDisplay
        display = PipelineDisplay()
        # update without start — should be a no-op / no crash
        display.update(1, status="running", detail="test")
        display.update(1, status="done", detail="finished")

    def test_valid_status_icons(self):
        from src.utils.progress import STATUS_ICONS
        for key in ("pending", "running", "done", "skipped", "failed"):
            assert key in STATUS_ICONS

    def test_all_phase_names_present(self):
        from src.utils.progress import PHASE_NAMES
        for phase in range(1, 8):
            assert phase in PHASE_NAMES
