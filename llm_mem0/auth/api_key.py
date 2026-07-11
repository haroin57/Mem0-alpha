"""Standard API-key authentication.

This is the fallback backend and the one path that works with any provider
mem0 itself supports (Anthropic, OpenAI, Gemini, ...) for mem0's *internal*
calls (``mem0_llm_config``). This library's own direct helper calls
(``complete`` — classification/extraction) are implemented for Anthropic and
OpenAI; other providers can still be used for mem0's internal calls by
passing ``provider=`` explicitly, but ``complete()`` will raise for them.

Set ``MEM0_LLM_PROVIDER`` + the provider's own API key env var
(``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ...). Without ``MEM0_LLM_PROVIDER``,
the provider is auto-detected from whichever key env var is set
(Anthropic first, then OpenAI).
"""

from __future__ import annotations

import logging
import os

from ..settings import DEFAULT_HELPER_MODEL, settings
from .base import AuthBackend

log = logging.getLogger(__name__)

_DEFAULT_MODELS = {
    "anthropic": DEFAULT_HELPER_MODEL,
    "openai": "gpt-5-mini",
}

_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


class ApiKeyAuth(AuthBackend):
    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider or self._detect_provider()
        if self.provider is None:
            raise RuntimeError(
                "No LLM API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
                "or MEM0_LLM_PROVIDER plus the matching key env var."
            )
        self._client = None

    @staticmethod
    def _detect_provider() -> str | None:
        if settings.MEM0_LLM_PROVIDER:
            return settings.MEM0_LLM_PROVIDER
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        return None

    def default_model(self) -> str:
        return _DEFAULT_MODELS.get(self.provider, settings.MEM0_LLM_MODEL)

    def mem0_llm_config(self, *, model: str) -> dict:
        config: dict = {"model": model, "temperature": 0.1, "max_tokens": 2000}
        key = self._api_key()
        if key:
            config["api_key"] = key
        return {"provider": self.provider, "config": config}

    def _api_key(self) -> str | None:
        key_env = _KEY_ENV.get(self.provider)
        return os.environ.get(key_env) if key_env else None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.provider not in _KEY_ENV:
            raise RuntimeError(
                f"llm_mem0's direct helper calls (classification/extraction) only "
                f"support provider='anthropic' or 'openai' natively; "
                f"provider={self.provider!r} can still be used for mem0's own "
                f"internal Memory.add()/search() calls via mem0_llm_config()."
            )
        key = self._api_key()
        if not key:
            # MEM0_LLM_PROVIDER can select a provider whose key env var is
            # unset; fail with a clear message instead of a raw KeyError.
            raise RuntimeError(
                f"provider={self.provider!r} selected but "
                f"{_KEY_ENV[self.provider]} is not set"
            )
        if self.provider == "anthropic":
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=key)
        else:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=key)
        return self._client

    async def aclose(self) -> None:
        """Close the cached async client (if any)."""
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.close()
            except Exception as exc:
                log.debug("ApiKeyAuth.aclose: %s", exc)

    async def complete(self, *, system: str, user_message: str, model: str, max_tokens: int) -> str:
        try:
            client = self._get_client()
        except Exception as exc:
            log.warning("ApiKeyAuth.complete: client init failed: %s", exc)
            return ""
        try:
            if self.provider == "anthropic":
                resp = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user_message}],
                )
                for block in getattr(resp, "content", []) or []:
                    text = getattr(block, "text", None)
                    if text:
                        return text
                return ""
            if self.provider == "openai":
                resp = await client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],
                )
                return resp.choices[0].message.content or ""
        except Exception as exc:
            log.warning("ApiKeyAuth.complete failed: %s", exc)
            return ""
        return ""
