"""
tests/test_inference.py — Unit tests for the inference pipeline module.

Tests cover:
  - Prompt builder for TinyLlama vs generic models
  - InferencePipeline initialization
  - Lazy loading (load called once)
  - Generate output structure
  - Adapter fallback when adapter dir is missing
  - unload() frees memory references
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.pipeline import _build_prompt, _is_tinyllama, InferencePipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestIsTinyllama:
    def test_tinyllama_detected(self):
        assert _is_tinyllama("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    def test_tinyllama_case_insensitive(self):
        assert _is_tinyllama("tinyllama-1.1b")

    def test_gpt2_not_tinyllama(self):
        assert not _is_tinyllama("gpt2")

    def test_bert_not_tinyllama(self):
        assert not _is_tinyllama("bert-base-uncased")


class TestBuildPrompt:
    def test_tinyllama_chat_format(self):
        prompt = _build_prompt("What is attention?", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        assert "<|system|>" in prompt
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt
        assert "What is attention?" in prompt

    def test_generic_format(self):
        prompt = _build_prompt("What is GPT?", "gpt2")
        assert "Question:" in prompt
        assert "Answer:" in prompt
        assert "What is GPT?" in prompt

    def test_generic_format_no_chat_markers(self):
        prompt = _build_prompt("Test?", "gpt2")
        assert "<|system|>" not in prompt


# ---------------------------------------------------------------------------
# InferencePipeline init
# ---------------------------------------------------------------------------

class TestInferencePipelineInit:
    def test_default_attributes(self, config):
        pipe = InferencePipeline(config)
        assert pipe.use_fine_tuned is False
        assert pipe._loaded is False
        assert pipe._model is None
        assert pipe._tokenizer is None

    def test_model_name_override(self, config):
        pipe = InferencePipeline(config, model_name_override="distilgpt2")
        assert pipe._effective_model_name == "distilgpt2"

    def test_effective_model_name_from_config(self, config):
        pipe = InferencePipeline(config)
        assert pipe._effective_model_name == config["model"]["base_model"]

    def test_fine_tuned_flag(self, config):
        pipe = InferencePipeline(config, use_fine_tuned=True)
        assert pipe.use_fine_tuned is True


# ---------------------------------------------------------------------------
# InferencePipeline.load — all heavy deps mocked
# ---------------------------------------------------------------------------

class TestInferencePipelineLoad:

    def _make_mock_tokenizer(self):
        tok = MagicMock()
        tok.pad_token = None
        tok.eos_token = "<eos>"
        tok.eos_token_id = 0
        return tok

    def _make_mock_model(self):
        model = MagicMock()
        model.device.type = "cpu"
        return model

    @patch("src.inference.pipeline.InferencePipeline._gpu_available", return_value=False)
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_load_sets_loaded_flag(self, mock_tok_cls, mock_model_cls, mock_gpu, config):
        mock_tok_cls.return_value = self._make_mock_tokenizer()
        mock_model_cls.return_value = self._make_mock_model()

        pipe = InferencePipeline(config)
        pipe.load()
        assert pipe._loaded is True

    @patch("src.inference.pipeline.InferencePipeline._gpu_available", return_value=False)
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_load_called_twice_loads_once(self, mock_tok_cls, mock_model_cls, mock_gpu, config):
        mock_tok_cls.return_value = self._make_mock_tokenizer()
        mock_model_cls.return_value = self._make_mock_model()

        pipe = InferencePipeline(config)
        pipe.load()
        pipe.load()  # second call should be no-op
        assert mock_model_cls.call_count == 1

    @patch("src.inference.pipeline.InferencePipeline._gpu_available", return_value=False)
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_load_sets_eos_as_pad_when_pad_missing(self, mock_tok_cls, mock_model_cls, mock_gpu, config):
        tok = self._make_mock_tokenizer()
        tok.pad_token = None
        mock_tok_cls.return_value = tok
        mock_model_cls.return_value = self._make_mock_model()

        pipe = InferencePipeline(config)
        pipe.load()
        assert tok.pad_token == tok.eos_token

    @patch("src.inference.pipeline.InferencePipeline._gpu_available", return_value=False)
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_fine_tuned_falls_back_when_adapter_missing(
        self, mock_tok_cls, mock_model_cls, mock_gpu, tmp_config
    ):
        """If adapter dir doesn't exist, use_fine_tuned should be set to False."""
        mock_tok_cls.return_value = self._make_mock_tokenizer()
        mock_model_cls.return_value = self._make_mock_model()

        # Ensure adapter dir does NOT exist
        adapter_path = Path(tmp_config["paths"]["models_adapter"])
        if adapter_path.exists():
            import shutil
            shutil.rmtree(adapter_path)

        pipe = InferencePipeline(tmp_config, use_fine_tuned=True)
        pipe.load()
        assert pipe.use_fine_tuned is False, (
            "Should fall back to base model when adapter dir is missing"
        )


# ---------------------------------------------------------------------------
# InferencePipeline.unload
# ---------------------------------------------------------------------------

class TestInferencePipelineUnload:
    def test_unload_clears_references(self, config):
        pipe = InferencePipeline(config)
        pipe._model = MagicMock()
        pipe._tokenizer = MagicMock()
        pipe._loaded = True

        pipe.unload()
        assert pipe._model is None
        assert pipe._tokenizer is None
        assert pipe._loaded is False


# ---------------------------------------------------------------------------
# InferencePipeline.generate — model mocked
# ---------------------------------------------------------------------------

class TestInferencePipelineGenerate:

    def _setup_loaded_pipe(self, config) -> InferencePipeline:
        """Return a pipe with mocked model/tokenizer already loaded."""
        import torch

        pipe = InferencePipeline(config)
        pipe._loaded = True

        tok = MagicMock()
        tok.eos_token_id = 0
        tok.decode.return_value = "The transformer is a model."
        # tokenizer call returns input_ids tensor
        tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        pipe._tokenizer = tok

        model = MagicMock()
        model.device.type = "cpu"
        # generate returns output_ids (input + 3 new tokens)
        model.generate.return_value = torch.tensor([[1, 2, 3, 10, 11, 12]])
        pipe._model = model

        return pipe

    def test_generate_returns_dict_keys(self, config):
        pipe = self._setup_loaded_pipe(config)
        result = pipe.generate("What is the transformer?")

        assert "response" in result
        assert "latency_s" in result
        assert "token_count" in result
        assert "input_tokens" in result
        assert "model_type" in result

    def test_generate_model_type_base(self, config):
        pipe = self._setup_loaded_pipe(config)
        result = pipe.generate("What is attention?")
        assert result["model_type"] == "base"

    def test_generate_token_count_non_negative(self, config):
        pipe = self._setup_loaded_pipe(config)
        result = pipe.generate("Test question")
        assert result["token_count"] >= 0

    def test_generate_latency_positive(self, config):
        pipe = self._setup_loaded_pipe(config)
        result = pipe.generate("Test question")
        assert result["latency_s"] >= 0
