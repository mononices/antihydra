"""Shared construction of model clients for the entry points."""

from __future__ import annotations

from typing import Any

from .llm import OpenRouterLLM, load_catalog
from .overseer import Overseer, OverseerConfig

DEFAULT_OVERSEER_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_OVERSEER_EFFORT = "max"


class Runtime:
    """Model clients and the price catalog, cached per (model, effort, params)."""

    def __init__(self, api_key: str, catalog: dict[str, Any] | None = None) -> None:
        if not api_key:
            raise ValueError("an OpenRouter API key is required")
        self.api_key = api_key
        self.catalog = catalog if catalog is not None else load_catalog()
        self._llms: dict[tuple, OpenRouterLLM] = {}
        self._compact_llms: dict[str, OpenRouterLLM] = {}
        self._overseer_llms: dict[tuple[str, str], OpenRouterLLM] = {}

    def llm(
        self,
        model: str,
        reasoning_effort: str = "max",
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> OpenRouterLLM:
        key = (model, reasoning_effort, temperature, max_tokens)
        if key not in self._llms:
            extra: dict[str, Any] = {}
            if temperature is not None:
                extra["temperature"] = temperature
            if max_tokens is not None:
                extra["max_tokens"] = max_tokens
            self._llms[key] = OpenRouterLLM(
                model,
                self.api_key,
                catalog=self.catalog,
                extra_body={"reasoning": {"effort": reasoning_effort}},
                **extra,
            )
        return self._llms[key]

    def compaction_llm(self, model: str) -> OpenRouterLLM:
        if model not in self._compact_llms:
            self._compact_llms[model] = OpenRouterLLM(
                model,
                self.api_key,
                temperature=0.0,
                max_tokens=1200,
                catalog=self.catalog,
            )
        return self._compact_llms[model]

    def overseer(self, config: OverseerConfig | None = None) -> Overseer:
        settings = config or OverseerConfig()
        key = (settings.model, settings.reasoning_effort)
        if key not in self._overseer_llms:
            self._overseer_llms[key] = OpenRouterLLM(
                settings.model,
                self.api_key,
                temperature=0.0,
                max_tokens=2000,
                catalog=self.catalog,
                extra_body={"reasoning": {"effort": settings.reasoning_effort}},
            )
        return Overseer(self._overseer_llms[key], settings)
