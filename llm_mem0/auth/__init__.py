"""Auto-detecting auth backend selection.

Order: an existing Claude Code CLI OAuth session (no separate API key
needed), then a standard API key env var for any provider mem0 supports.
"""

from __future__ import annotations

import logging
import os
import sys

from .anthropic_cli import CREDENTIALS_PATH, AnthropicCliAuth
from .api_key import ApiKeyAuth
from .base import AuthBackend

log = logging.getLogger(__name__)

_backend: AuthBackend | None = None


def get_auth_backend() -> AuthBackend:
    """Return the process-wide auth backend, auto-detected on first call.

    Tries an existing Claude Code CLI OAuth session first — the credentials
    file on Linux, or the login Keychain on macOS (where the CLI stores them
    there instead of the file) — and falls back to a standard API key.
    """
    global _backend
    if _backend is not None:
        return _backend
    # On macOS the CLI keeps credentials in the Keychain, so the file may not
    # exist even when a session is available; let the backend's constructor
    # probe both sources and raise only if neither has a usable token.
    if os.path.isfile(CREDENTIALS_PATH) or sys.platform == "darwin":
        try:
            _backend = AnthropicCliAuth()
            log.info("llm_mem0: using Claude Code CLI OAuth session")
            return _backend
        except Exception as exc:
            log.warning("Claude Code CLI session not usable (%s); falling back to API key", exc)
    _backend = ApiKeyAuth()
    log.info("llm_mem0: using API key auth (provider=%s)", _backend.provider)
    return _backend


def set_auth_backend(backend: AuthBackend) -> None:
    """Explicitly install an auth backend, bypassing auto-detection.

    Host applications with their own credential management (an existing
    OAuth manager, a billing-routing wrapper, a provider fallback chain)
    implement :class:`AuthBackend` and inject it here — before the first
    LLM-touching operation — instead of letting ``get_auth_backend`` probe
    the environment. Injection wins over auto-detection because
    ``get_auth_backend`` returns any cached backend first.
    """
    global _backend
    _backend = backend
    log.info("llm_mem0: auth backend injected (%s)", type(backend).__name__)


def reset_auth_backend() -> None:
    """Clear the cached backend. Mainly useful for tests."""
    global _backend
    _backend = None


async def aclose_auth_backend() -> None:
    """Close the cached backend's network clients and clear the singleton.

    For embedded hosts / tests that build and tear down event loops; a
    long-lived process can simply exit without calling this.
    """
    global _backend
    backend, _backend = _backend, None
    if backend is not None:
        await backend.aclose()


__all__ = [
    "AuthBackend",
    "AnthropicCliAuth",
    "ApiKeyAuth",
    "aclose_auth_backend",
    "get_auth_backend",
    "reset_auth_backend",
    "set_auth_backend",
]
