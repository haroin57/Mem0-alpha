"""Abstract interface for pluggable LLM authentication backends.

llm_mem0 makes two kinds of LLM calls:

1. mem0's own internal calls (fact extraction/dedup during ``Memory.add()``,
   embedding-based search) go through mem0's ``Memory.from_config()`` and
   use mem0's own multi-provider ``LlmFactory``. A backend only needs to
   supply the ``{"provider": ..., "config": {...}}`` block for this.

2. This library's own helper calls (metadata classification, self-fact
   extraction, query rewrite/rerank) are small, single-turn completions
   made directly against a model, outside mem0's ``Memory`` object. A
   backend supplies a ``complete()`` coroutine for these so callers never
   depend on a specific provider's SDK shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class AuthBackend(ABC):
    """A provider-agnostic handle for both mem0-internal and direct LLM calls."""

    @abstractmethod
    async def complete(
        self, *, system: str, user_message: str, model: str, max_tokens: int,
    ) -> str:
        """Run a single-turn completion and return the raw text response.

        Returns ``""`` on any failure — callers treat these helper calls as
        best-effort and must not raise.
        """

    @abstractmethod
    def mem0_llm_config(self, *, model: str) -> dict:
        """Return the mem0 ``{"provider": ..., "config": {...}}`` block used
        by ``Memory.from_config()`` for mem0's own internal LLM calls."""

    @abstractmethod
    def default_model(self) -> str:
        """A small/fast model name suitable for this backend's helper calls."""

    async def aclose(self) -> None:
        """Release any cached network clients. Default: nothing to release.

        Long-lived hosts don't need this; embedded hosts and tests that tear
        down event loops should call :func:`llm_mem0.auth.aclose_auth_backend`
        (which delegates here) to avoid unclosed-session warnings.
        """
