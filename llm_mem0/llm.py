"""Shared plumbing for the library's own single-shot helper LLM calls.

Every helper stage (extraction, classification, dedup judgment, conflict
classification, query rewrite, HyDE, rerank) follows the same dance: fetch
the auth backend, escape braces in untrusted text bound for a
``str.format`` template, run one completion, strip the Markdown fence the
model may wrap its JSON in, parse, and fail open on any error. Nine call
sites used to hand-roll it; this module is the single implementation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)


def escape_braces(text: str, *, limit: int | None = None) -> str:
    """Escape ``{``/``}`` so untrusted text is inert inside a format template.

    ``limit`` caps the length first (the templates all cap their slots), so
    call sites read as one operation instead of a slice + two replaces.
    """
    if limit is not None:
        text = text[:limit]
    return text.replace("{", "{{").replace("}", "}}")


def strip_json_fence(text: str) -> str:
    """Remove a Markdown ```json fence around an LLM JSON reply, if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()
    return text


async def complete_json(
    *,
    system: str,
    user_message: str,
    model: str,
    max_tokens: int,
    caller: str = "llm",
) -> Any:
    """One completion, parsed as JSON. Returns ``None`` on ANY failure.

    Fail-open contract: backend errors, empty replies, and parse failures
    all log a WARNING tagged with ``caller`` and return ``None``, so no
    helper stage can take down the ingestion or recall path.
    """
    try:
        from .auth import get_auth_backend
        backend = get_auth_backend()
        text = strip_json_fence(await backend.complete(
            system=system,
            user_message=user_message,
            model=model,
            max_tokens=max_tokens,
        ))
        if not text:
            log.warning("%s: empty completion (fail-open)", caller)
            return None
        return json.loads(text)
    except Exception as exc:
        log.warning("%s: LLM JSON call failed (fail-open): %s", caller, exc)
        return None
