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

from .base import AuthBackend

log = logging.getLogger(__name__)

_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-5-mini",
}


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
        configured = os.environ.get("MEM0_LLM_PROVIDER")
        if configured:
            return configured
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        return None

    def default_model(self) -> str:
        return _DEFAULT_MODELS.get(self.provider, os.environ.get("MEM0_LLM_MODEL", ""))

    def mem0_llm_config(self, *, model: str) -> dict:
        config: dict = {"model": model, "temperature": 0.1, "max_tokens": 2000}
        key_env = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(self.provider)
        if key_env and os.environ.get(key_env):
            config["api_key"] = os.environ[key_env]
        return {"provider": self.provider, "config": config}

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.provider == "anthropic":
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif self.provider == "openai":
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            raise RuntimeError(
                f"llm_mem0's direct helper calls (classification/extraction) only "
                f"support provider='anthropic' or 'openai' natively; "
                f"provider={self.provider!r} can still be used for mem0's own "
                f"internal Memory.add()/search() calls via mem0_llm_config()."
            )
        return self._client

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
