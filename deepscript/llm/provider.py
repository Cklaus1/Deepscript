"""LLM provider — Claude, OpenAI, Ollama, vLLM, SGLang, NIM. With retry + async + rate limiting."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from deepscript.config.settings import LLMConfig
from deepscript.llm.cost_tracker import CostTracker

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Rate limits per provider (requests per minute)
PROVIDER_RATE_LIMITS = {
    "nim": 35,       # NIM free tier ~40, stay under
    "ollama": 0,     # Local — no limit
    "vllm": 0,       # Local — no limit
    "sglang": 0,     # Local — no limit
    "claude": 0,     # Handled by Anthropic SDK
    "openai": 0,     # Handled by OpenAI SDK
    "none": 0,
}


class _RateLimiter:
    """Thread-safe rate limiter. Shared per provider."""

    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        self._lock = threading.Lock()
        self._last_request = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self._last_request = time.monotonic()


# One rate limiter per provider — shared across all LLMProvider instances
_limiters: dict[str, _RateLimiter] = {}
_limiters_lock = threading.Lock()


def _get_limiter(provider: str) -> _RateLimiter:
    with _limiters_lock:
        if provider not in _limiters:
            rpm = PROVIDER_RATE_LIMITS.get(provider, 0)
            _limiters[provider] = _RateLimiter(rpm)
        return _limiters[provider]

LOCAL_BASE_URLS = {
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "sglang": "http://localhost:30000/v1",
    "nim": "https://integrate.api.nvidia.com/v1",
}

OPENAI_COMPAT_PROVIDERS = {"openai", "ollama", "vllm", "sglang", "nim"}
NO_KEY_PROVIDERS = {"ollama", "vllm", "sglang"}

# Exception type names that are transient (worth retrying)
TRANSIENT_ERRORS = {"APIConnectionError", "RateLimitError", "InternalServerError"}
# Exception type names that are permanent (don't retry)
PERMANENT_ERRORS = {"AuthenticationError", "BadRequestError", "PermissionDeniedError"}


def _is_transient(exc: Exception) -> bool:
    """Check if an exception is transient and worth retrying."""
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    name = type(exc).__name__
    if name in TRANSIENT_ERRORS:
        return True
    # Check for HTTP 429/5xx status codes on API errors
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True
    return False


class LLMProvider:
    """Unified LLM provider with retry, async, and cost tracking."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.cost_tracker = CostTracker(budget_limit=config.budget_per_month)
        # Override provider default rate limit if config specifies one
        if config.rate_limit_rpm > 0:
            with _limiters_lock:
                _limiters[config.provider] = _RateLimiter(config.rate_limit_rpm)
        self._client: Any = None
        self._async_client: Any = None

    @staticmethod
    def create(config: LLMConfig) -> Optional["LLMProvider"]:
        """Create an LLM provider if configured and credentials available."""
        if config.provider == "none":
            return None

        if config.provider == "claude":
            if not (config.api_key or os.environ.get("ANTHROPIC_API_KEY")):
                logger.warning("ANTHROPIC_API_KEY not set, falling back to rule-based")
                return None
            return LLMProvider(config)

        if config.provider == "openai":
            if not (config.api_key or os.environ.get("OPENAI_API_KEY")):
                logger.warning("OPENAI_API_KEY not set, falling back to rule-based")
                return None
            return LLMProvider(config)

        if config.provider in ("ollama", "vllm", "sglang"):
            return LLMProvider(config)

        if config.provider == "nim":
            if not (config.api_key or os.environ.get("NVIDIA_API_KEY")):
                logger.warning("NVIDIA_API_KEY not set, falling back to rule-based")
                return None
            return LLMProvider(config)

        logger.warning("LLM provider '%s' not supported", config.provider)
        return None

    # --- Client initialization ---

    def _get_client(self) -> Any:
        """Lazy-initialize the sync LLM client."""
        if self._client is not None:
            return self._client
        self._client = self._build_client(async_mode=False)
        return self._client

    def _get_async_client(self) -> Any:
        """Lazy-initialize the async LLM client."""
        if self._async_client is not None:
            return self._async_client
        self._async_client = self._build_client(async_mode=True)
        return self._async_client

    def _build_client(self, async_mode: bool = False) -> Any:
        """Build sync or async client based on provider."""
        if self.config.provider == "claude":
            import anthropic
            kwargs: dict[str, Any] = {}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            return anthropic.AsyncAnthropic(**kwargs) if async_mode else anthropic.Anthropic(**kwargs)

        elif self.config.provider in OPENAI_COMPAT_PROVIDERS:
            import openai
            base_url = self.config.base_url or LOCAL_BASE_URLS.get(self.config.provider)
            api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
            if self.config.provider in NO_KEY_PROVIDERS and not api_key:
                api_key = "not-needed"
            if self.config.provider == "nim" and not api_key:
                api_key = os.environ.get("NVIDIA_API_KEY", "not-needed")
            kwargs = {}
            if base_url:
                kwargs["base_url"] = base_url
            if api_key:
                kwargs["api_key"] = api_key
            return openai.AsyncOpenAI(**kwargs) if async_mode else openai.OpenAI(**kwargs)

        return None

    # --- Sync completion with retry ---

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        """Send a prompt to the LLM with retry on transient failures.

        Returns None if all attempts fail (caller falls back to rule-based).
        """
        if self.cost_tracker.budget_exceeded:
            logger.warning("Budget exceeded — skipping LLM call")
            return None

        max_retries = self.config.max_retries
        last_error: Exception | None = None

        for attempt in range(max_retries):
            _get_limiter(self.config.provider).wait()
            try:
                client = self._get_client()
                start = time.monotonic()

                if self.config.provider == "claude":
                    result = self._complete_anthropic(client, prompt, system, max_tokens)
                else:
                    result = self._complete_openai_compat(client, prompt, system, max_tokens)

                elapsed_ms = int((time.monotonic() - start) * 1000)
                if self.cost_tracker.entries:
                    self.cost_tracker.entries[-1].latency_ms = elapsed_ms
                    self.cost_tracker.entries[-1].provider = self.config.provider

                return result

            except Exception as e:
                last_error = e
                if _is_transient(e) and attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.info("LLM call failed (attempt %d/%d), retrying in %ds: %s",
                                attempt + 1, max_retries, wait, e)
                    time.sleep(wait)
                    continue

                # Permanent error or final attempt
                exc_type = type(e).__name__
                if exc_type in PERMANENT_ERRORS:
                    logger.warning("LLM permanent error (%s): %s", self.config.provider, e)
                elif _is_transient(e):
                    logger.warning("LLM failed after %d retries (%s): %s",
                                   max_retries, self.config.provider, e)
                else:
                    logger.exception("Unexpected LLM error (%s): %s", self.config.provider, e)
                return None

        return None

    # --- Async completion with retry ---

    async def complete_async(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        """Async version of complete() with retry on transient failures."""
        import asyncio

        if self.cost_tracker.budget_exceeded:
            return None

        max_retries = self.config.max_retries
        for attempt in range(max_retries):
            # Rate limit (run in thread to not block event loop)
            await asyncio.to_thread(_get_limiter(self.config.provider).wait)
            try:
                client = self._get_async_client()
                start = time.monotonic()

                if self.config.provider == "claude":
                    result = await self._complete_anthropic_async(client, prompt, system, max_tokens)
                else:
                    result = await self._complete_openai_compat_async(client, prompt, system, max_tokens)

                elapsed_ms = int((time.monotonic() - start) * 1000)
                if self.cost_tracker.entries:
                    self.cost_tracker.entries[-1].latency_ms = elapsed_ms
                    self.cost_tracker.entries[-1].provider = self.config.provider

                return result

            except Exception as e:
                if _is_transient(e) and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.info("Async LLM retry %d/%d in %ds: %s", attempt + 1, max_retries, wait, e)
                    await asyncio.sleep(wait)
                    continue
                logger.warning("Async LLM failed (%s): %s", self.config.provider, e)
                return None

        return None

    async def complete_json_async(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        """Async version of complete_json()."""
        text = await self.complete_async(prompt, system=system, max_tokens=max_tokens)
        if text is None:
            return None
        return self._parse_json(text)

    # --- Provider-specific implementations ---

    def _complete_anthropic(self, client: Any, prompt: str, system: str | None, max_tokens: int | None) -> str:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens or self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        if hasattr(response, "usage") and self.config.cost_tracking:
            self.cost_tracker.record(self.config.model, response.usage.input_tokens, response.usage.output_tokens)
        return response.content[0].text

    async def _complete_anthropic_async(self, client: Any, prompt: str, system: str | None, max_tokens: int | None) -> str:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens or self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = await client.messages.create(**kwargs)
        if hasattr(response, "usage") and self.config.cost_tracking:
            self.cost_tracker.record(self.config.model, response.usage.input_tokens, response.usage.output_tokens)
        return response.content[0].text

    def _complete_openai_compat(self, client: Any, prompt: str, system: str | None, max_tokens: int | None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=self.config.model, messages=messages,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        if hasattr(response, "usage") and response.usage and self.config.cost_tracking:
            self.cost_tracker.record(
                self.config.model,
                getattr(response.usage, "prompt_tokens", 0) or 0,
                getattr(response.usage, "completion_tokens", 0) or 0,
            )
        return response.choices[0].message.content

    async def _complete_openai_compat_async(self, client: Any, prompt: str, system: str | None, max_tokens: int | None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await client.chat.completions.create(
            model=self.config.model, messages=messages,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        if hasattr(response, "usage") and response.usage and self.config.cost_tracking:
            self.cost_tracker.record(
                self.config.model,
                getattr(response.usage, "prompt_tokens", 0) or 0,
                getattr(response.usage, "completion_tokens", 0) or 0,
            )
        return response.choices[0].message.content

    # --- JSON parsing ---

    def complete_json(self, prompt: str, system: str | None = None, max_tokens: int | None = None) -> dict[str, Any] | None:
        """Send a prompt and parse the response as JSON."""
        text = self.complete(prompt, system=system, max_tokens=max_tokens)
        if text is None:
            return None
        return self._parse_json(text)

    def _parse_json(self, text: str) -> dict[str, Any] | None:
        """Extract and parse JSON from LLM response (handles markdown wrapping)."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON: %s...", text[:200])
            return None

    # --- Prompt templates (cached) ---

    _template_cache: dict[str, str] = {}

    def render_prompt(self, template_name: str, **kwargs: str) -> str:
        """Load a prompt template (cached) and fill in placeholders."""
        if template_name not in self._template_cache:
            template_path = PROMPTS_DIR / f"{template_name}.txt"
            self._template_cache[template_name] = template_path.read_text(encoding="utf-8")
        return self._template_cache[template_name].format(**kwargs)
