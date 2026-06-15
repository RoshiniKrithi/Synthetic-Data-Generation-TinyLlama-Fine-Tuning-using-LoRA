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


class InferencePipeline:
    """Reusable inference wrapper for TinyLlama (base or PEFT fine-tuned).

    Args:
        config: Full project config dict.
        use_fine_tuned: If True load the LoRA adapter; otherwise load base model.
    """

    def __init__(self, config: dict, use_fine_tuned: bool = False) -> None:
        self.cfg_inf = config["inference"]
        self.cfg_model = config["model"]
        self.paths = config["paths"]
        self.use_fine_tuned = use_fine_tuned
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the model and tokenizer into memory (lazy — call once)."""
        if self._loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_name = self.cfg_model["base_model"]

        logger.info("Loading tokenizer from %s", base_name)
        self._tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
        self._tokenizer.pad_token = self._tokenizer.eos_token

        device_map = "auto" if self._gpu_available() else None
        dtype = torch.float32

        if self.use_fine_tuned:
            adapter_dir = self.paths["models_adapter"]
            if not Path(adapter_dir).exists():
                logger.warning("Adapter not found at %s — falling back to base model.", adapter_dir)
                self.use_fine_tuned = False
            else:
                from peft import PeftModel
                logger.info("Loading base model + LoRA adapter …")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
                )
                self._model = PeftModel.from_pretrained(base_model, adapter_dir)
                self._loaded = True
                logger.info("Fine-tuned model loaded.")
                return

        logger.info("Loading base model: %s", base_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
        )
        self._loaded = True
        logger.info("Base model loaded.")

    def generate(self, question: str) -> dict:
        """Generate an answer for *question*.

        Args:
            question: The input question string.

        Returns:
            Dict with keys ``response``, ``latency_s``, ``token_count``,
            ``input_tokens``, and ``model_type``.
        """
        if not self._loaded:
            self.load()

        import torch

        prompt = (
            f"<|system|>\nYou are a helpful, accurate assistant.\n"
            f"<|user|>\n{question}\n"
            f"<|assistant|>\n"
        )

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"]

        # Move to model device
        if hasattr(self._model, "device") and self._model.device.type != "cpu":
            input_ids = input_ids.to(self._model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=self.cfg_inf["max_new_tokens"],
                temperature=self.cfg_inf["temperature"],
                top_p=self.cfg_inf["top_p"],
                do_sample=self.cfg_inf["do_sample"],
                repetition_penalty=self.cfg_inf["repetition_penalty"],
                pad_token_id=self._tokenizer.eos_token_id,
            )
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
        """Return True if a CUDA GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
