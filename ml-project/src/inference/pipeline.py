"""
Phase 6 — Inference Pipeline

Wraps HuggingFace generation into a reusable object that returns the
response, latency, and token count for any question.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TINYLLAMA_MODELS = {"tinyllama"}


def _is_tinyllama(model_name: str) -> bool:
    return "tinyllama" in model_name.lower()


def _build_prompt(question: str, model_name: str) -> str:
    """Return a prompt string appropriate for the model family."""
    if _is_tinyllama(model_name):
        return (
            f"<|system|>\nYou are a helpful, accurate assistant.\n"
            f"<|user|>\n{question}\n"
            f"<|assistant|>\n"
        )
    # Generic causal-LM format (GPT-2, etc.) — must match training format
    return f"Question: {question}\nAnswer:"


class InferencePipeline:
    """Reusable inference wrapper for causal language models.

    Args:
        config: Full project config dict.
        use_fine_tuned: If True load the LoRA adapter; otherwise load base model.
        model_name_override: If set, use this HF model ID instead of config's base_model.
    """

    def __init__(
        self,
        config: dict,
        use_fine_tuned: bool = False,
        model_name_override: str | None = None,
    ) -> None:
        self.cfg_inf = config["inference"]
        self.cfg_model = config["model"]
        self.paths = config["paths"]
        self.use_fine_tuned = use_fine_tuned
        self._model_name_override = model_name_override
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def _effective_model_name(self) -> str:
        return self._model_name_override or self.cfg_model["base_model"]

    def load(self) -> None:
        """Load the model and tokenizer into memory (lazy — call once)."""
        if self._loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_name = self._effective_model_name

        logger.info("Loading tokenizer from %s", base_name)
        self._tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        device_map = "auto" if self._gpu_available() else None
        dtype = torch.float32

        if self.use_fine_tuned:
            adapter_dir = self.paths["models_adapter"]
            if not Path(adapter_dir).exists():
                logger.warning(
                    "Adapter not found at %s — falling back to base model.", adapter_dir
                )
                self.use_fine_tuned = False
            else:
                from peft import PeftModel
                logger.info("Loading base model + LoRA adapter …")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_name, dtype=dtype, device_map=device_map, trust_remote_code=True
                )
                self._model = PeftModel.from_pretrained(base_model, adapter_dir)
                self._loaded = True
                logger.info("Fine-tuned model loaded.")
                return

        logger.info("Loading base model: %s", base_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            base_name, dtype=dtype, device_map=device_map, trust_remote_code=True
        )
        self._loaded = True
        logger.info("Base model loaded.")

    def generate(self, question: str) -> dict:
        """Generate an answer for *question*.

        Returns:
            Dict with keys ``response``, ``latency_s``, ``token_count``,
            ``input_tokens``, and ``model_type``.
        """
        if not self._loaded:
            self.load()

        import torch

        model_name = self._effective_model_name
        prompt = _build_prompt(question, model_name)

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"]

        if hasattr(self._model, "device") and self._model.device.type != "cpu":
            input_ids = input_ids.to(self._model.device)

        gen_kwargs: dict = {
            "max_new_tokens": self.cfg_inf["max_new_tokens"],
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if self.cfg_inf.get("do_sample", True):
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(self.cfg_inf["temperature"])
            gen_kwargs["top_p"] = float(self.cfg_inf["top_p"])
        else:
            gen_kwargs["do_sample"] = False
        if self.cfg_inf.get("repetition_penalty"):
            gen_kwargs["repetition_penalty"] = float(self.cfg_inf["repetition_penalty"])

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(input_ids, **gen_kwargs)
        latency = time.perf_counter() - t0

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "response": response,
            "latency_s": round(latency, 3),
            "token_count": int(new_tokens.shape[0]),
            "input_tokens": int(input_ids.shape[-1]),
            "model_type": "fine_tuned" if self.use_fine_tuned else "base",
        }

    def unload(self) -> None:
        """Release model from memory."""
        self._model = None
        self._tokenizer = None
        self._loaded = False
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gpu_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
