"""Hybrid LLM router — local for PII-sensitive work, cloud for bulk categorization.

Routing rules:
  - PDF extraction / raw text analysis → llama.cpp (local, PII stays on machine)
  - Transaction categorization (scrubbed) → OpenRouter (fast, cheap) or local fallback
  - Receipt line-item extraction → llama.cpp (receipts may contain PII)

Usage:
    router = LLMRouter()
    model, base_url, api_key = router.resolve("categorize")  # → OpenRouter if available
    model, base_url, api_key = router.resolve("extract")      # → always local
"""
import logging
from enum import Enum

from app.agents.llm_provider import LLMProviderRegistry

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    EXTRACT = "extract"           # Raw PDF → transactions (PII present)
    RECEIPT = "receipt"           # Receipt → line items (PII present)
    CATEGORIZE = "categorize"    # Scrubbed transactions → categories (no PII)
    ANALYZE = "analyze"          # Scrubbed aggregates → insights (no PII)


# Tasks that MUST stay local (contain PII)
_LOCAL_ONLY: frozenset[TaskType] = frozenset({TaskType.EXTRACT, TaskType.RECEIPT})

# Preferred cloud providers for non-PII tasks, in priority order
_CLOUD_PREFERENCES: list[tuple[str, str]] = [
    ("openrouter", "openrouter/google/gemini-2.5-flash"),
    ("gemini", "gemini/gemini-2.5-flash"),
    ("openai", "openai/gpt-4o-mini"),
]

# Local providers, in priority order
_LOCAL_PREFERENCES: list[tuple[str, str]] = [
    ("llamacpp", "openai/local-model"),
    ("lm-studio", "openai/qwen2.5-14b-instruct"),
    ("ollama", "ollama/llama3.2"),
]


class LLMRouter:
    def __init__(self, registry: LLMProviderRegistry | None = None):
        self._registry = registry or LLMProviderRegistry()

    def resolve(self, task: TaskType | str) -> tuple[str, str, str]:
        """Return (litellm_model, base_url, api_key) for the given task type.

        PII-sensitive tasks always route to local LLM.
        Non-PII tasks try cloud first (faster/cheaper), fall back to local.
        """
        task = TaskType(task) if isinstance(task, str) else task

        if task in _LOCAL_ONLY:
            return self._resolve_local()

        # Non-PII: try cloud first, fall back to local
        try:
            return self._resolve_cloud()
        except ValueError:
            logger.info("No cloud provider available, falling back to local for %s", task)
            return self._resolve_local()

    def _resolve_cloud(self) -> tuple[str, str, str]:
        """Try cloud providers in preference order."""
        for provider_id, model_id in _CLOUD_PREFERENCES:
            try:
                return self._registry.resolve(provider_id, model_id)
            except ValueError:
                continue
        raise ValueError("No cloud LLM provider available")

    def _resolve_local(self) -> tuple[str, str, str]:
        """Try local providers in preference order."""
        for provider_id, model_id in _LOCAL_PREFERENCES:
            try:
                return self._registry.resolve(provider_id, model_id)
            except ValueError:
                continue
        raise ValueError("No local LLM provider available (llama.cpp / LM Studio / Ollama)")

    def describe_route(self, task: TaskType | str) -> str:
        """Human-readable description of where a task would route."""
        task = TaskType(task) if isinstance(task, str) else task
        try:
            model, base_url, _ = self.resolve(task)
            location = "LOCAL" if task in _LOCAL_ONLY else "CLOUD (scrubbed)"
            return f"{task.value} → {model} [{location}] {base_url or '(default)'}"
        except ValueError as e:
            return f"{task.value} → UNAVAILABLE: {e}"
