"""
tests/conftest.py — Shared pytest fixtures for the ML pipeline test suite.
"""
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make sure the project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Minimal config reused across all test modules
MINIMAL_CONFIG: dict = {
    "project": {
        "name": "Test Project",
        "version": "0.0.1",
        "description": "Test config",
    },
    "scraper": {
        "wikipedia_url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "chunk_size": 100,
        "chunk_overlap": 10,
        "max_chunks": 5,
        "retry_attempts": 2,
        "retry_delay": 0,
    },
    "generator": {
        "ollama_model": "mistral",
        "ollama_base_url": "http://localhost:11434",
        "qa_pairs_per_chunk": 2,
        "max_qa_pairs": 20,
        "temperature": 0.7,
        "question_types": ["factual", "definition"],
        "retry_attempts": 1,
    },
    "model": {
        "base_model": "gpt2",
        "max_length": 64,
        "device": "cpu",
    },
    "lora": {
        "r": 2,
        "alpha": 4,
        "dropout": 0.0,
        "target_modules": ["c_attn"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    "training": {
        "epochs": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-3,
        "warmup_steps": 0,
        "save_steps": 100,
        "eval_steps": 100,
        "logging_steps": 1,
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,
        "fp16": False,
        "bf16": False,
        "dataloader_num_workers": 0,
        "seed": 42,
    },
    "evaluation": {
        "metrics": ["exact_match", "bleu"],
        "test_split": 0.2,
        "max_samples": 5,
        "bertscore_model": "distilbert-base-uncased",
    },
    "inference": {
        "max_new_tokens": 16,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": False,
        "repetition_penalty": 1.0,
    },
    "gradio": {
        "host": "127.0.0.1",
        "port": 7860,
        "share": False,
        "title": "Test UI",
        "description": "Test",
        "demo_model": "gpt2",
    },
    "logging": {
        "level": "WARNING",
        "format": "%(message)s",
        "file": "logs/test.log",
    },
    "paths": {
        "data_raw": "data/raw",
        "data_processed": "data/processed",
        "data_synthetic": "data/synthetic",
        "models_base": "models/base",
        "models_adapter": "models/adapter",
        "models_merged": "models/merged",
        "reports": "reports",
        "charts": "reports/charts",
        "logs": "logs",
    },
}


@pytest.fixture
def config() -> dict:
    """Return the minimal project config dict."""
    return MINIMAL_CONFIG.copy()


@pytest.fixture
def tmp_config(tmp_path: Path) -> dict:
    """Return a config where all paths point to tmp_path subdirs."""
    import copy
    cfg = copy.deepcopy(MINIMAL_CONFIG)
    for key in cfg["paths"]:
        cfg["paths"][key] = str(tmp_path / cfg["paths"][key])
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    cfg["logging"]["file"] = str(tmp_path / "logs" / "test.log")
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return cfg


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Return a small list of synthetic chunk dicts."""
    return [
        {
            "id": i,
            "text": (
                f"The transformer model is a type of deep learning architecture. "
                f"It was introduced by Vaswani et al. in 2017. "
                f"Chunk number {i} contains additional context about attention mechanisms."
            ),
            "word_count": 35,
        }
        for i in range(5)
    ]


@pytest.fixture
def sample_qa_pairs() -> list[dict]:
    """Return a small list of synthetic Q&A pairs."""
    return [
        {
            "question": "What is the transformer architecture?",
            "answer": "The transformer is a deep learning model that uses self-attention mechanisms.",
            "type": "factual",
            "chunk_id": 0,
        },
        {
            "question": "Who introduced the transformer model?",
            "answer": "Vaswani et al. introduced it in the paper 'Attention Is All You Need' in 2017.",
            "type": "factual",
            "chunk_id": 0,
        },
        {
            "question": "What is self-attention?",
            "answer": "Self-attention is a mechanism that allows each position in a sequence to attend to all others.",
            "type": "definition",
            "chunk_id": 1,
        },
        {
            "question": "How does multi-head attention differ from single-head attention?",
            "answer": "Multi-head attention applies attention in parallel h times with different learned projections.",
            "type": "reasoning",
            "chunk_id": 1,
        },
        {
            "question": "What is positional encoding in transformers?",
            "answer": "Positional encoding injects information about the position of tokens into the embeddings.",
            "type": "definition",
            "chunk_id": 2,
        },
    ]


@pytest.fixture
def mock_inference_pipeline(sample_qa_pairs):
    """Return a MagicMock that looks like an InferencePipeline."""
    pipe = MagicMock()
    pipe.cfg_inf = {
        "max_new_tokens": 16,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": False,
        "repetition_penalty": 1.0,
    }

    def _generate(question: str) -> dict:
        # Return a deterministic dummy response
        return {
            "response": f"Answer to: {question[:30]}",
            "latency_s": 0.01,
            "token_count": 5,
            "input_tokens": 10,
            "model_type": "base",
        }

    pipe.generate.side_effect = _generate
    return pipe
