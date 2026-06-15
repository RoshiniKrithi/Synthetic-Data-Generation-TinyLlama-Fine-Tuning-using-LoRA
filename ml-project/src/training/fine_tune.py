"""
Phase 4 — LoRA Fine-Tuning

Fine-tunes TinyLlama on the synthetic Q&A dataset using PEFT + LoRA.
CPU-safe with optional GPU acceleration.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _format_prompt(example: dict) -> str:
    """Format a Q&A pair into a chat-style training prompt.

    Args:
        example: Dict with ``question`` and ``answer`` keys.

    Returns:
        Formatted string for causal language model training.
    """
    return (
        f"<|system|>\nYou are a helpful assistant.\n"
        f"<|user|>\n{example['question']}\n"
        f"<|assistant|>\n{example['answer']}</s>"
    )


class LoRAFineTuner:
    """Manage the full fine-tuning lifecycle: dataset → model → LoRA → train → save.

    Args:
        config: Full project config dict.
    """

    def __init__(self, config: dict) -> None:
        self.cfg_model = config["model"]
        self.cfg_lora = config["lora"]
        self.cfg_train = config["training"]
        self.paths = config["paths"]
        self.base_model_name: str = self.cfg_model["base_model"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, qa_pairs: list[dict]) -> str:
        """Execute the fine-tuning pipeline.

        Args:
            qa_pairs: List of Q&A pair dicts.

        Returns:
            Path to the saved adapter directory.
        """
        logger.info("Starting LoRA fine-tuning on %d Q&A pairs …", len(qa_pairs))
        self._check_dependencies()

        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        device = self._detect_device()
        logger.info("Using device: %s", device)

        # ------ Tokenizer ------
        logger.info("Loading tokenizer: %s", self.base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # ------ Base model ------
        logger.info("Loading base model …")
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch.float32,
        }
        if device == "cuda":
            load_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(self.base_model_name, **load_kwargs)
        model.config.use_cache = False

        # ------ LoRA config ------
        lora_config = LoraConfig(
            r=self.cfg_lora["r"],
            lora_alpha=self.cfg_lora["alpha"],
            target_modules=self.cfg_lora["target_modules"],
            lora_dropout=self.cfg_lora["dropout"],
            bias=self.cfg_lora["bias"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # ------ Dataset ------
        prompts = [_format_prompt(p) for p in qa_pairs]
        raw_ds = Dataset.from_dict({"text": prompts})
        split = raw_ds.train_test_split(test_size=0.1, seed=self.cfg_train["seed"])

        max_length = self.cfg_model["max_length"]

        def tokenize(batch: dict) -> dict:
            out = tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            out["labels"] = out["input_ids"].copy()
            return out

        tokenized = split.map(tokenize, batched=True, remove_columns=["text"])

        # ------ Training args ------
        adapter_dir = str(Path(self.paths["models_adapter"]).resolve())
        training_args = TrainingArguments(
            output_dir=adapter_dir,
            num_train_epochs=self.cfg_train["epochs"],
            per_device_train_batch_size=self.cfg_train["batch_size"],
            gradient_accumulation_steps=self.cfg_train["gradient_accumulation_steps"],
            learning_rate=self.cfg_train["learning_rate"],
            warmup_steps=self.cfg_train["warmup_steps"],
            save_steps=self.cfg_train["save_steps"],
            eval_steps=self.cfg_train["eval_steps"],
            logging_steps=self.cfg_train["logging_steps"],
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            max_grad_norm=self.cfg_train["max_grad_norm"],
            weight_decay=self.cfg_train["weight_decay"],
            fp16=device == "cuda" and self.cfg_train.get("fp16", False),
            no_cuda=device == "cpu",
            seed=self.cfg_train["seed"],
            report_to="none",
            dataloader_num_workers=0,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["test"],
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )

        logger.info("Training started …")
        trainer.train()
        logger.info("Training complete.")

        # ------ Save adapter ------
        Path(adapter_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        logger.info("Adapter saved → %s", adapter_dir)
        return adapter_dir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        """Return the best available device string."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @staticmethod
    def _check_dependencies() -> None:
        """Raise ImportError early if required packages are missing."""
        missing = []
        for pkg in ["torch", "transformers", "peft", "datasets"]:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            raise ImportError(f"Missing packages: {missing}. Run: pip install {' '.join(missing)}")
