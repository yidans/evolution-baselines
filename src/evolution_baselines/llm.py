"""Model clients for the baseline arms: Azure OpenAI, scripted (tests), and a budget ledger."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ._io import load_project_env


class BudgetExhausted(RuntimeError):
    """Raised when the arm has used every model call it was allowed."""


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    seconds: float
    cached: bool = False


class LLMClient(Protocol):
    model: str

    def complete(self, *, system: str, user: str, tag: str) -> LLMResponse: ...


class ScriptedClient:
    """Deterministic stand-in for tests: replies in order, cycling when exhausted."""

    model = "scripted"

    def __init__(self, responses: list[str], *, cycle: bool = True) -> None:
        if not responses:
            raise ValueError("ScriptedClient needs at least one response")
        self._responses = list(responses)
        self._cycle = cycle
        self.calls = 0
        self.prompts: list[dict[str, str]] = []

    def complete(self, *, system: str, user: str, tag: str) -> LLMResponse:
        if self.calls >= len(self._responses) and not self._cycle:
            raise BudgetExhausted("scripted responses exhausted")
        text = self._responses[self.calls % len(self._responses)]
        self.calls += 1
        self.prompts.append({"tag": tag, "system": system, "user": user})
        return LLMResponse(text=text, model=self.model, prompt_tokens=len(user) // 4,
                           completion_tokens=len(text) // 4, seconds=0.0)


class AzureOpenAIClient:
    """Chat completions against the OpenAI-compatible ``/openai/v1/`` Azure endpoint."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str,
        max_output_tokens: int = 16_000,
        reasoning_effort: str | None = None,
        timeout_seconds: float = 900.0,
        max_retries: int = 0,
    ) -> None:
        from openai import OpenAI

        if not endpoint.startswith("https://"):
            raise ValueError("model endpoint must be https")
        self.model = model
        self.max_output_tokens = int(max_output_tokens)
        self.reasoning_effort = reasoning_effort
        if max_retries != 0:
            raise ValueError("SDK retries must be disabled so every API attempt is charged")
        self._client = OpenAI(base_url=endpoint.rstrip("/") + "/", api_key=api_key,
                              timeout=timeout_seconds, max_retries=max_retries)

    @classmethod
    def from_env(cls, *, model: str | None = None, **kwargs: Any) -> "AzureOpenAIClient":
        load_project_env()
        endpoint = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip()
        api_key = (os.environ.get("AZURE_OPENAI_API_KEY") or "").strip()
        resolved = (model or os.environ.get("AZURE_OPENAI_MODEL")
                    or os.environ.get("AZURE_OPENAI_CHAT_MODEL") or "").strip()
        missing = [name for name, value in (("AZURE_OPENAI_ENDPOINT", endpoint),
                                            ("AZURE_OPENAI_API_KEY", api_key),
                                            ("AZURE_OPENAI_MODEL", resolved)) if not value]
        if missing:
            raise RuntimeError("missing environment: " + ", ".join(missing))
        return cls(endpoint=endpoint, api_key=api_key, model=resolved, **kwargs)

    def complete(self, *, system: str, user: str, tag: str) -> LLMResponse:
        start = time.perf_counter()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort
        response = self._client.chat.completions.create(**request)
        text = (response.choices[0].message.content or "") if response.choices else ""
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text, model=str(getattr(response, "model", self.model)),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            seconds=time.perf_counter() - start)


class LedgerClient:
    """Budget accounting + on-disk replay cache around any client.

    Every logical call (cache hit or physical request) counts against ``max_calls`` so a
    replay from cache walks the same iteration path as the paid run.  The ledger
    (``calls.jsonl``) records prompts, responses and token usage for audit.
    """

    def __init__(self, inner: LLMClient, out_dir: Path, *, max_calls: int,
                 cache_path: Path | None = None) -> None:
        self.inner = inner
        self.model = inner.model
        self.request_settings = {
            "reasoning_effort": getattr(inner, "reasoning_effort", None),
            "max_output_tokens": getattr(inner, "max_output_tokens", None),
        }
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls = int(max_calls)
        if self.max_calls < 0:
            raise ValueError("max_calls must be nonnegative")
        self.ledger_path = self.out_dir / "calls.jsonl"
        if self.ledger_path.exists():
            raise FileExistsError(f"run already has calls: {self.out_dir}; use a new output directory")
        self.cache_path = Path(cache_path) if cache_path else self.out_dir / "llm_cache.jsonl"
        self.calls_used = 0
        self.physical_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.failed_calls = 0
        self.unknown_usage_calls = 0
        self._cache: dict[str, dict[str, Any]] = {}
        if self.cache_path.is_file():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    item = json.loads(line)
                    self._cache[item["key"]] = item

    @property
    def remaining(self) -> int:
        return self.max_calls - self.calls_used

    def _key(self, system: str, user: str) -> str:
        return hashlib.sha256(f"{self.model}\x00{system}\x00{user}".encode("utf-8")).hexdigest()

    def complete(self, *, system: str, user: str, tag: str) -> LLMResponse:
        if self.calls_used >= self.max_calls:
            raise BudgetExhausted(f"model-call budget exhausted ({self.max_calls})")
        key = self._key(system, user)
        hit = self._cache.get(key)
        if hit is not None and "request_settings" in hit and hit["request_settings"] != self.request_settings:
            hit = None
        if hit is None and isinstance(self.inner, _ReplayOnlyClient):
            raise BudgetExhausted(f"replay client has no cached response for {tag}")
        self.calls_used += 1
        if hit is not None:
            response = LLMResponse(text=hit["text"], model=hit.get("model", self.model),
                                   prompt_tokens=hit.get("prompt_tokens"),
                                   completion_tokens=hit.get("completion_tokens"), seconds=0.0,
                                   cached=True)
        else:
            self.physical_calls += 1
            start = time.perf_counter()
            try:
                response = self.inner.complete(system=system, user=user, tag=tag)
            except Exception as exc:
                self.failed_calls += 1
                self.unknown_usage_calls += 1
                with self.ledger_path.open("a") as handle:
                    handle.write(json.dumps({
                        "call": self.calls_used, "tag": tag, "cached": False,
                        "model": self.model, "status": "failed",
                        "error_type": type(exc).__name__,
                        "prompt_tokens": None, "completion_tokens": None,
                        "seconds": time.perf_counter() - start,
                    }) + "\n")
                raise
            item = {"key": key, "text": response.text, "model": response.model,
                    "request_model": self.model,
                    "request_settings": self.request_settings,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens}
            self._cache[key] = item
            with self.cache_path.open("a") as handle:
                handle.write(json.dumps(item) + "\n")
        if response.prompt_tokens is None or response.completion_tokens is None:
            self.unknown_usage_calls += 1
        self.prompt_tokens += int(response.prompt_tokens or 0)
        self.completion_tokens += int(response.completion_tokens or 0)
        with self.ledger_path.open("a") as handle:
            handle.write(json.dumps({
                "call": self.calls_used, "tag": tag, "cached": response.cached, "model": response.model,
                "status": "completed",
                "request_settings": self.request_settings,
                "prompt_sha256": key, "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens, "seconds": response.seconds,
                "system": system, "user": user, "response": response.text,
            }) + "\n")
        return response

    def usage(self) -> dict[str, Any]:
        return {"model": self.model, "max_calls": self.max_calls, "calls_used": self.calls_used,
                "physical_calls": self.physical_calls, "cached_calls": self.calls_used - self.physical_calls,
                "failed_calls": self.failed_calls, "unknown_usage_calls": self.unknown_usage_calls,
                "client_kind": type(self.inner).__name__,
                "reasoning_effort": getattr(self.inner, "reasoning_effort", None),
                "max_output_tokens": getattr(self.inner, "max_output_tokens", None),
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}


def build_client(kind: str, *, out_dir: Path, max_calls: int, model: str | None = None,
                 reasoning_effort: str | None = None, scripted: list[str] | None = None,
                 cache_path: Path | None = None, max_output_tokens: int = 16_000) -> LedgerClient:
    if kind == "azure":
        inner: LLMClient = AzureOpenAIClient.from_env(model=model, reasoning_effort=reasoning_effort,
                                                     max_output_tokens=max_output_tokens)
    elif kind == "scripted":
        inner = ScriptedClient(scripted or [])
    elif kind == "replay":
        cache_path = Path(cache_path) if cache_path else Path(out_dir) / "llm_cache.jsonl"
        if not cache_path.is_file():
            raise ValueError("replay requires an existing --cache file")
        if model is None:
            models = {json.loads(line).get("request_model")
                      for line in cache_path.read_text().splitlines() if line.strip()}
            if len(models) != 1 or not all(isinstance(value, str) and value for value in models):
                raise ValueError("set --model to the original request model for this legacy or mixed cache")
            model = models.pop()
        settings = [json.loads(line).get("request_settings")
                    for line in cache_path.read_text().splitlines() if line.strip()]
        if settings and any(setting != settings[0] for setting in settings):
            raise ValueError("replay requires a cache with one set of request settings")
        recorded_settings = settings[0] if settings else None
        if reasoning_effort is not None and recorded_settings and reasoning_effort != recorded_settings["reasoning_effort"]:
            raise ValueError("replay reasoning effort differs from the recorded request")
        inner = _ReplayOnlyClient(model)
        if recorded_settings:
            inner.reasoning_effort = recorded_settings["reasoning_effort"]
            inner.max_output_tokens = recorded_settings["max_output_tokens"]
    else:
        raise ValueError(f"unknown client kind {kind!r}")
    return LedgerClient(inner, out_dir, max_calls=max_calls, cache_path=cache_path)


class _ReplayOnlyClient:
    """Refuses physical calls: every prompt must already be in the ledger cache."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.reasoning_effort: str | None = None
        self.max_output_tokens: int | None = None

    def complete(self, *, system: str, user: str, tag: str) -> LLMResponse:
        raise BudgetExhausted(f"replay client has no cached response for {tag}")


__all__ = ["AzureOpenAIClient", "BudgetExhausted", "LLMClient", "LLMResponse", "LedgerClient",
           "ScriptedClient", "build_client"]
