"""Japanese morphological analysis helpers (Phase A-1 follow-up).

Wraps Janome so the rest of the Mem0 stack — both the BM25 keyword
index and the vector smart_search — can operate on content words
rather than raw character soup.

Why this matters:
- Japanese has no whitespace word boundaries, so a "natural sentence"
  query is one giant CJK chunk. trigram FTS5 can't break it apart on
  its own, so phrase-matching the whole chunk returns 0 hits.
- Embedding tokenizers (OpenAI / Cohere) do split CJK via their own
  byte-pair model, but the resulting vector still weights stop-word
  particles like ``の は が で と``. Pre-extracting content words
  gives the smart_search expand step a clean keyword variant to fuse
  alongside the raw query.

Public API:
    tokenize_keywords(text) -> list[str]
        Content-word surface forms in source order (no particles, no
        auxiliary verbs, no symbols). Empty list on empty input.
    tokenize_for_index(text) -> str
        Whitespace-joined surface forms. Used as BM25 facts_fts.text.
    extract_query_keywords(query, max_tokens=8) -> str
        Same surface forms, capped + space-joined, suitable for handing
        to ``mem.search`` as an extra query variant. Empty string on
        empty/no-content input.

The Janome ``Tokenizer`` instance is built lazily and cached for the
process lifetime; instantiation takes ~300ms and ~10MB resident memory.
Tests can call ``_reset_tokenizer_for_test()`` between runs to force
re-instantiation.
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable

log = logging.getLogger(__name__)


_tokenizer = None
_tokenizer_lock = threading.Lock()


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    with _tokenizer_lock:
        if _tokenizer is not None:
            return _tokenizer
        try:
            from janome.tokenizer import Tokenizer
            _tokenizer = Tokenizer()
        except Exception as exc:
            log.warning("janome unavailable, morpho disabled: %s", exc)
            _tokenizer = False  # sentinel so we don't retry on every call
        return _tokenizer


def _reset_tokenizer_for_test() -> None:
    """Test hook — force re-instantiation on next call."""
    global _tokenizer
    _tokenizer = None


# Content-word POS prefixes we keep. Janome returns part_of_speech as
# comma-separated levels; we match on the leading category. Symbols /
# particles / fillers are deliberately excluded.
_KEEP_POS = ("名詞", "動詞", "形容詞", "副詞")

# Sub-POS we drop even within an accepted top-level (e.g. 名詞,非自立 is
# a dependent noun like "こと" / "もの" that's near-stop-word).
_DROP_SUBPOS = ("非自立", "数",)


def _is_content_token(token) -> bool:
    pos_parts = token.part_of_speech.split(",")
    if not pos_parts:
        return False
    if pos_parts[0] not in _KEEP_POS:
        return False
    if len(pos_parts) >= 2 and pos_parts[1] in _DROP_SUBPOS:
        return False
    return True


def tokenize_keywords(text: str) -> list[str]:
    """Surface forms of content words. Empty list if Janome unavailable
    or input is empty."""
    if not text or not text.strip():
        return []
    t = _get_tokenizer()
    if not t:
        return []
    out: list[str] = []
    try:
        for token in t.tokenize(text):
            if not _is_content_token(token):
                continue
            surface = (token.surface or "").strip()
            if not surface:
                continue
            # Default-dictionary Janome splits new proper nouns
            # ("上伊那ぼたん" → "上伊那"/"ぼ"/"たん") so a 1-char surface
            # tends to be either a verb-stem fragment ("し" / "い") or
            # an over-segmented noun byte. Both are noise in BM25 and
            # in the keyword query variant; drop them. ASCII 1-char
            # tokens like "C" would also drop here — accepted tradeoff
            # since C as a search query is meaningless without context.
            if len(surface) < 2:
                continue
            out.append(surface)
    except Exception as exc:
        log.warning("morpho tokenize failed (fail-open): %s", exc)
        return []
    return out


def tokenize_for_index(text: str) -> str:
    """Whitespace-joined surface forms for FTS5 storage.

    Falls back to the raw text when Janome is unavailable / extraction
    yields nothing, so we never write an empty string to the index for
    a non-empty source fact.
    """
    kws = tokenize_keywords(text)
    if kws:
        return " ".join(kws)
    return (text or "").strip()


def extract_query_keywords(query: str, *, max_tokens: int = 8) -> str:
    """Whitespace-joined keyword variant of ``query``. Returns empty
    string when no content words are found — caller is expected to skip
    that variant.

    Cap is per Mem0 search latency: a 50-keyword query inflates the
    embedding payload without improving recall. 8 captures the salient
    nouns in most conversational messages.
    """
    kws = tokenize_keywords(query)
    if not kws:
        return ""
    # Stable de-dupe while preserving source order.
    seen: set[str] = set()
    out: list[str] = []
    for k in kws:
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
        if len(out) >= max_tokens:
            break
    return " ".join(out)
