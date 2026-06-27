"""
Shared utility helpers for the ML pipeline.
"""
from __future__ import annotations

import json
import logging
import os

# Redirect HF Cache to D: drive due to C: drive running out of space (0 bytes free)
os.environ["HF_HOME"] = "d:/Synthetic Data Generation + TinyLlama Fine-Tuning using LoRA/Asset-Manager/Asset-Manager/hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
from pathlib import Path
from typing import Any

import yaml


def setup_logging(log_file: str = "logs/project.log", level: str = "INFO") -> logging.Logger:
    """Configure root logger with console + file handlers.

    Args:
        log_file: Path to the log file.
        level: Logging level string (e.g. "INFO", "DEBUG").

    Returns:
        Configured root logger.
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load a YAML config file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict[str, Any]) -> None:
    """Create all project directories from the paths section of config.

    Args:
        config: Full project config dict (must contain a 'paths' key).
    """
    for key, path in config.get("paths", {}).items():
        Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Serialize *data* to a JSON file.

    Args:
        data: JSON-serialisable object.
        path: Output file path (parent dirs are created automatically).
        indent: JSON indentation level.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Any:
    """Load a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON object.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Convert a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        String like "1h 23m 45s".
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def get_device() -> str:
    """Detect the best available compute device (CUDA > MPS > CPU).

    Returns:
        Device string suitable for PyTorch / HuggingFace.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
