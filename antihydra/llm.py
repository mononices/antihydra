"""OpenRouter chat client and model price catalog (standard library only)."""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

API_ROOT = "https://openrouter.ai/api/v1"

REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}


class LLMError(RuntimeError):
    """Raised when the model API cannot fulfil a request."""


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any] | None
    raw_arguments: str | None = None


@dataclass
class Reply:
    content: str | None
    tool_calls: list[ToolCallRequest]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    finish_reason: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class ModelPrice:
    id: str
    prompt: float  # USD per prompt token
    completion: float  # USD per completion token
    context_length: int | None = None
    input_modalities: tuple[str, ...] = ()


def normalize_model(model: str) -> str:
    """Allow ``openrouter/<vendor>/<model>`` and ``<vendor>/<model>`` alike."""
    prefix = "openrouter/"
    return model[len(prefix) :] if model.startswith(prefix) else model


def fetch_catalog(*, timeout: float = 30) -> dict[str, ModelPrice]:
    """Fetch the model list and prices from the public ``/models`` endpoint."""
    request = urllib.request.Request(
        f"{API_ROOT}/models", headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8", "replace"))
    catalog: dict[str, ModelPrice] = {}
    for entry in payload.get("data", []):
        model_id = entry.get("id")
        if not model_id:
            continue
        pricing = entry.get("pricing") or {}
        try:
            prompt = float(pricing.get("prompt") or 0.0)
            completion = float(pricing.get("completion") or 0.0)
        except (TypeError, ValueError):
            prompt = completion = 0.0
        catalog[model_id] = ModelPrice(
            id=model_id,
            prompt=max(prompt, 0.0),
            completion=max(completion, 0.0),
            context_length=entry.get("context_length"),
            input_modalities=tuple(
                str(item) for item in ((entry.get("architecture") or {}).get("input_modalities") or [])
            ),
        )
    return catalog


def load_catalog(
    cache_dir: str | Path = ".cache",
    *,
    max_age: float = 7 * 24 * 3600,
    refresh: bool = False,
) -> dict[str, ModelPrice]:
    """Return the price catalog, using a local cache with graceful fallback."""
    cache_path = Path(cache_dir) / "openrouter_models.json"
    cached: dict[str, ModelPrice] = {}
    fetched_at = 0.0
    if cache_path.is_file():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            fetched_at = float(raw.get("fetched_at", 0))
            cached = {
                model_id: ModelPrice(
                    id=model_id,
                    prompt=float(entry.get("prompt", 0) or 0),
                    completion=float(entry.get("completion", 0) or 0),
                    context_length=entry.get("context_length"),
                    input_modalities=tuple(entry.get("input_modalities") or ()),
                )
                for model_id, entry in (raw.get("models") or {}).items()
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            cached = {}
    if cached and not refresh and time.time() - fetched_at < max_age:
        return cached
    try:
        catalog = fetch_catalog()
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return cached
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "fetched_at": time.time(),
                "models": {
                    model_id: {
                        "prompt": price.prompt,
                        "completion": price.completion,
                        "context_length": price.context_length,
                        "input_modalities": list(price.input_modalities),
                    }
                    for model_id, price in catalog.items()
                },
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return catalog


class OpenRouterLLM:
    """Thin chat-completions client with tool calling, retries and cost tracking."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str = API_ROOT,
        timeout: float = 600,
        temperature: float | None = 0.7,
        max_tokens: int | None = 4096,
        catalog: dict[str, ModelPrice] | None = None,
        extra_body: dict[str, Any] | None = None,
        retries: int = 3,
    ) -> None:
        if not api_key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        self.model = normalize_model(model)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.catalog = catalog or {}
        self.extra_body = extra_body or {}
        self.retries = retries
        self._warned_unknown_price = False

    @property
    def context_length(self) -> int | None:
        """Model context window from the price catalog, when known."""
        price = self.catalog.get(self.model)
        return price.context_length if price else None

    def supports_images(self) -> bool | None:
        """True/False when the catalog reports modalities, None when unknown."""
        price = self.catalog.get(self.model)
        if price is None or not price.input_modalities:
            return None
        return "image" in price.input_modalities

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "antihydra-solver",
        }

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers=self._headers(),
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8", "replace"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace") if exc.fp else ""
                last_error = LLMError(
                    f"HTTP {exc.code} from OpenRouter: {detail[:400]}"
                )
                if exc.code not in RETRYABLE_STATUS or attempt >= self.retries:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = LLMError(f"request to OpenRouter failed: {exc}")
                if attempt >= self.retries:
                    raise last_error from exc
            time.sleep(min(30.0, (2**attempt) + random.random()))
        raise last_error or LLMError("request failed")

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        price = self.catalog.get(self.model)
        if price is None:
            if not self._warned_unknown_price:
                self._warned_unknown_price = True
                print(
                    f"[warn] no price entry for {self.model}; cost tracking stays at 0.00"
                )
            return 0.0
        return prompt_tokens * price.prompt + completion_tokens * price.completion

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Reply:
        """One chat-completions call."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "usage": {"include": True},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        temp = self.temperature if temperature is None else temperature
        if temp is not None:
            payload["temperature"] = temp
        tokens = self.max_tokens if max_tokens is None else max_tokens
        if tokens:
            payload["max_tokens"] = tokens
        payload.update(self.extra_body)

        response = self._post(payload)
        choices = response.get("choices") or []
        if not choices:
            raise LLMError(f"no choices in response: {json.dumps(response)[:400]}")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):  # some providers return content parts
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )

        tool_calls: list[ToolCallRequest] = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            raw = function.get("arguments")
            arguments: dict[str, Any] | None = None
            if isinstance(raw, dict):
                arguments = raw
            elif isinstance(raw, str):
                try:
                    arguments = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError:
                    arguments = None
            tool_calls.append(
                ToolCallRequest(
                    id=str(call.get("id") or f"call_{index}"),
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                    raw_arguments=(
                        raw if isinstance(raw, str) else json.dumps(raw or {})
                    ),
                )
            )

        usage = response.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        reported_cost = usage.get("cost")
        cost = (
            float(reported_cost)
            if reported_cost is not None
            else self._estimate_cost(prompt_tokens, completion_tokens)
        )
        return Reply(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            finish_reason=choice.get("finish_reason"),
            model=response.get("model"),
        )
