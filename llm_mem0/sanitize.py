"""Prompt-injection sentinel regex + sanitize helpers.

Retrieved memories and history hits are *untrusted* text: a fact ingested
earlier could contain markers that, when echoed back into a prompt, close a
fenced block or impersonate a trusted prompt-scaffold header (a second-order
prompt-injection). These helpers strip such markers defensively.

Unicode bypass mitigation: callers matching against adversarial text should
`normalize_for_sentinel(text)` first (or use `apply_sentinel_filters`, which
bundles normalization + sanitization). This folds Unicode lookalikes (NFKC)
and strips zero-width characters that are otherwise invisible to the patterns.
"""

import re
import unicodedata

__all__ = [
    "SENTINEL_RE",
    "POWER_PHRASE_RE",
    "sanitize_untrusted_text",
    "normalize_for_sentinel",
    "apply_sentinel_filters",
]

# Zero-width / invisible characters that can split keywords to evade matching.
_ZERO_WIDTH_CHARS = (
    "​",  # zero-width space
    "‌",  # zero-width non-joiner
    "‍",  # zero-width joiner
    "⁠",  # word joiner
    "﻿",  # zero-width no-break space (BOM)
    "­",  # soft hyphen
    "͏",  # combining grapheme joiner
)


def normalize_for_sentinel(text: str) -> str:
    """Normalize untrusted text before sentinel/power-phrase matching.

    Applies NFKC Unicode normalization (folds Latin-Extended lookalikes and
    other compatibility equivalents to their canonical base forms) and strips
    zero-width invisible characters that can split keywords.
    """
    text = unicodedata.normalize("NFKC", text)
    for ch in _ZERO_WIDTH_CHARS:
        text = text.replace(ch, "")
    return text


# Sentinel labels a trusted prompt scaffold might use around retrieved context.
# The default set here mirrors the fence/header markers this library emits in
# format.py (``[... memories]`` / ``[End of ...]`` / ``<retrieved_memories>``).
# If your own prompt scaffold uses different sentinel labels, extend this
# pattern (or sanitize with your own regex) so a stored fact cannot forge one.
SENTINEL_RE = re.compile(
    r"\[(?:End of|Memories?|Long-term memory|Attached images|Current message|"
    r"Trusted context|untrusted_user_input|trusted_user_input|History|"
    r"End of history)[^\]\n]*\]"
    r"|</?retrieved_memories[^>]*>"
    r"|</?history_search[^>]*>"
    r"|</?untrusted_user_input[^>]*>"
    r"|</?trusted_user_input[^>]*>",
    re.IGNORECASE,
)

# Power-grab phrases — patterns that try to impersonate trusted instructions
# ("ignore the previous instructions", "system prompt", "root access", ...).
POWER_PHRASE_RE = re.compile(
    r"(bypass[_\- ]permissions|all[_\- ]permissions|system[_\- ]prompt|"
    r"root[_\- ]access|ignore (?:the )?(?:previous|above)|"
    r"disregard (?:the )?(?:previous|above)|"
    r"</untrusted_user_input>|</trusted_user_input>)",
    re.IGNORECASE,
)


def sanitize_untrusted_text(text: str) -> str:
    """Remove prompt-scaffold sentinel labels and obvious power-grab phrases.

    Idempotent — running it twice on already-sanitized text returns the same
    result, so layered defenses can call it freely.

    Note: does NOT normalize Unicode. Callers that may receive adversarial
    input should use `apply_sentinel_filters` instead.
    """
    if not text:
        return text
    text = SENTINEL_RE.sub("[redacted-marker]", text)
    text = POWER_PHRASE_RE.sub("[redacted]", text)
    return text


def apply_sentinel_filters(text: str) -> str:
    """Normalize then sanitize untrusted text in one call, gaining
    Unicode-bypass protection."""
    if not text:
        return text
    return sanitize_untrusted_text(normalize_for_sentinel(text))
