"""llm_mem0 — provider-agnostic long-term memory for LLM agents.

A thin, opinionated layer over `mem0` that adds: a self-fact extraction gate
(one small-model call turns a conversation turn into clean, deduplicated
facts about the user), hybrid vector+BM25 retrieval with query rewrite and
rerank, an entity/relation graph, and prompt-injection-safe formatting of
retrieved memories.

Authentication is pluggable (see `llm_mem0.auth`): it can reuse an existing
Claude Code CLI OAuth session so no separate API key is needed, or fall back
to a standard API key for any provider mem0 supports.

Public API:
  client   — memory store singleton (also low-level get_all/delete)
  ingest   — add_memories (extraction gate + fallback)
  search   — search / search_multi / search_smart + rewrite/rerank
  extract  — extract_facts_for_self + _classify_for_metadata
  dedup    — two-stage dedup (cosine + LLM judgment)
  format   — format_memories_for_prompt / format_history_for_prompt
"""

from .client import (
    MEM0_RELEVANCE_MAX_DISTANCE,
    MEM0_RELEVANCE_THRESHOLD,  # back-compat alias; see client.py
    delete_memory,
    get_all_memories,
)
from .extract import extract_facts_for_self
from .format import (
    format_history_for_prompt,
    format_memories_for_prompt,
)
from .ingest import add_memories
from .search import (
    search_memories,
    search_memories_multi,
    search_memories_smart,
    should_use_memory_llm_mode,
)

__all__ = [
    "MEM0_RELEVANCE_MAX_DISTANCE",
    "MEM0_RELEVANCE_THRESHOLD",
    "add_memories",
    "delete_memory",
    "extract_facts_for_self",
    "format_history_for_prompt",
    "format_memories_for_prompt",
    "get_all_memories",
    "search_memories",
    "search_memories_multi",
    "search_memories_smart",
    "should_use_memory_llm_mode",
]
