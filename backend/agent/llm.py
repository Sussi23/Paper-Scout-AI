"""LLM access: a small Protocol plus the Gemini implementation (throttle, retry/backoff).

Function calling is manual: automatic function calling is disabled and the model's `Content`
objects are returned untouched so thought signatures survive the round trip.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

import httpx
from google import genai
from google.genai import errors, types

from backend.agent.run_logger import redact_secrets

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 503, 504}
MAX_BACKOFF_SECONDS = 30.0


class LLMError(Exception):
    """Unrecoverable LLM failure. `code` is LLM_QUOTA, LLM_UNAVAILABLE or LLM_ERROR."""

    def __init__(self, code: str, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass
class LLMResponse:
    """One model turn. `content` is the original object from the API (never rebuilt)."""

    content: types.Content
    text: str = ""
    function_calls: List[types.FunctionCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None

    @classmethod
    def from_content(
        cls,
        content: types.Content,
        usage: Optional[Dict[str, int]] = None,
        finish_reason: Optional[str] = None,
    ) -> "LLMResponse":
        """Derive text and function calls from a model Content."""
        parts = content.parts or []
        text = "".join(p.text for p in parts if p.text and not p.thought)
        calls = [p.function_call for p in parts if p.function_call is not None]
        return cls(content=content, text=text, function_calls=calls, usage=usage or {}, finish_reason=finish_reason)


class LLMClient(Protocol):
    """What the agent loop needs from a model. Tests replace it with FakeLLM."""

    def generate(
        self,
        contents: Sequence[types.Content],
        tools: Sequence[types.Tool],
        system_instruction: str,
    ) -> LLMResponse:
        """Return the next model turn or raise LLMError."""


class GeminiClient:
    """Gemini via google-genai with min-interval throttling and bounded retry/backoff."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        temperature: Optional[float] = None,
        min_interval_seconds: float = 2.0,
        max_retries: int = 3,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        secrets: Iterable[str] = (),
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._min_interval = min_interval_seconds
        self._max_retries = max_retries
        self._sleep = sleep
        self._clock = clock
        self._secrets = [s for s in [api_key, *secrets] if s]
        self._last_call: Optional[float] = None
        self._client = client or genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=120_000))

    def _throttle(self) -> None:
        if self._last_call is not None:
            wait = self._min_interval - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = self._clock()

    def _config(self, tools: Sequence[types.Tool], system_instruction: str) -> types.GenerateContentConfig:
        kwargs: Dict[str, Any] = dict(system_instruction=system_instruction)
        if tools:  # plain text calls (e.g. the non-agent baseline) carry no tool configuration
            kwargs.update(
                tools=list(tools),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.AUTO)
                ),
            )
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        return types.GenerateContentConfig(**kwargs)

    def _safe(self, text: str) -> str:
        return redact_secrets(text, self._secrets)[:400]

    def generate(
        self,
        contents: Sequence[types.Content],
        tools: Sequence[types.Tool],
        system_instruction: str,
    ) -> LLMResponse:
        config = self._config(tools, system_instruction)
        last: Optional[LLMError] = None
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=list(contents), config=config
                )
                candidate = resp.candidates[0] if resp.candidates else None
                if candidate is None or candidate.content is None or not candidate.content.parts:
                    reason = getattr(getattr(candidate, "finish_reason", None), "name", None)
                    last = LLMError("LLM_UNAVAILABLE", f"Gemini returned an empty response (finish_reason={reason}).")
                else:
                    return self._to_response(resp, candidate)
            except errors.APIError as exc:
                status = exc.code
                text = self._safe(str(getattr(exc, "message", "") or exc))
                if status in RETRYABLE_STATUS:
                    code = "LLM_QUOTA" if status == 429 else "LLM_UNAVAILABLE"
                    last = LLMError(code, f"Gemini HTTP {status}: {text}", status)
                else:
                    raise LLMError("LLM_ERROR", f"Gemini HTTP {status}: {text}", status) from None
            except (httpx.HTTPError, TimeoutError, ConnectionError) as exc:
                last = LLMError("LLM_UNAVAILABLE", f"Gemini network error: {type(exc).__name__}")
            if attempt < self._max_retries:
                delay = min(4.0 * (2 ** attempt), MAX_BACKOFF_SECONDS)
                logger.warning("Gemini %s, retry %d/%d in %.0fs", last.code, attempt + 1, self._max_retries, delay)
                self._sleep(delay)
        assert last is not None
        raise last

    @staticmethod
    def _to_response(resp: Any, candidate: Any) -> LLMResponse:
        usage_meta = getattr(resp, "usage_metadata", None)
        usage: Dict[str, int] = {}
        if usage_meta is not None:
            for name in ("prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count"):
                value = getattr(usage_meta, name, None)
                if isinstance(value, int):
                    usage[name] = value
        finish = getattr(candidate.finish_reason, "name", None) or (
            str(candidate.finish_reason) if candidate.finish_reason else None
        )
        return LLMResponse.from_content(candidate.content, usage=usage, finish_reason=finish)
