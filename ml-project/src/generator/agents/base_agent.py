"""
src/generator/agents/base_agent.py — Abstract base class for all pipeline agents.

All agents (Generator, Critic, Refiner) share:
  - Ollama client construction
  - JSON-safe generation with retry + back-off
  - Structured logging

Subclasses only implement ``_build_prompt()`` and ``_parse_response()``.
"""
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all pipeline agents.

    Args:
        config: Full project config dict.
        role: Human-readable name for this agent (used in logs).
    """

    def __init__(self, config: dict, role: str) -> None:
        self.role = role
        self.cfg_gen = config.get("generator", {})
        self.cfg_agent = config.get("agents", {})
        self.model: str = self.cfg_agent.get("model", self.cfg_gen.get("ollama_model", "qwen2.5:1.5b"))
        self.base_url: str = self.cfg_gen.get("ollama_base_url", "http://127.0.0.1:11434")
        self.temperature: float = float(self.cfg_agent.get("temperature", self.cfg_gen.get("temperature", 0.7)))
        self.retry_attempts: int = int(self.cfg_agent.get("retry_attempts", self.cfg_gen.get("retry_attempts", 3)))
        self._client = None

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_prompt(self, *args: Any, **kwargs: Any) -> str:
        """Construct the full prompt string for this agent."""

    @abstractmethod
    def _parse_response(self, raw: str, *args: Any, **kwargs: Any) -> Any:
        """Parse the model's raw string response into a structured object."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _call(self, prompt: str) -> str | None:
        """Send *prompt* to Ollama and return the raw response text.

        Retries with exponential back-off on failure.

        Args:
            prompt: Fully-formed prompt string.

        Returns:
            Raw response string from the model, or ``None`` on failure.
        """
        client = self._get_client()
        if client is None:
            return None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                resp = client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={"temperature": self.temperature},
                )
                return resp.get("response", "")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] Ollama attempt %d/%d failed: %s",
                    self.role, attempt, self.retry_attempts, exc,
                )
                if attempt < self.retry_attempts:
                    time.sleep(2 ** attempt)
        return None

    def _get_client(self):
        """Lazy-init and cache the Ollama client."""
        if self._client is not None:
            return self._client
        try:
            import ollama as _ollama
            self._client = _ollama.Client(host=self.base_url)
            return self._client
        except ImportError:
            logger.error("ollama package not installed. Run: pip install ollama")
            return None

    # ------------------------------------------------------------------
    # JSON parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_object(text: str) -> dict | None:
        """Extract the first JSON *object* from *text*.

        Tries three strategies:
        1. Regex to find ``{...}`` block.
        2. Parse entire text as JSON.
        3. Return ``None``.

        Args:
            text: Raw model output string.

        Returns:
            Parsed dict or ``None``.
        """
        # Strategy 1: find JSON object via regex
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 2: entire response
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        return None

    @staticmethod
    def _extract_json_array(text: str) -> list | None:
        """Extract the first JSON *array* from *text*.

        Args:
            text: Raw model output string.

        Returns:
            Parsed list or ``None``.
        """
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        try:
            data = json.loads(text.strip())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        return None
