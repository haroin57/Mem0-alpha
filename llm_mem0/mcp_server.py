"""MCP server wrapper for llm_mem0 (Mem0α).

Exposes the library's public memory API as Model Context Protocol tools so
any MCP client (Claude Code, Claude Desktop, Cursor, ...) can use Mem0α as a
long-term memory backend without writing Python.

Run::

    pip install "mem0-alpha[mcp]"
    mem0-alpha-mcp                    # stdio transport

Claude Code / Claude Desktop config example::

    {
      "mcpServers": {
        "mem0-alpha": {
          "command": "mem0-alpha-mcp",
          "env": {"MEM0_DEFAULT_USER_ID": "you"}
        }
      }
    }

Configuration (vector store address, models, auth backend) is inherited from
``llm_mem0.settings`` env vars — see the README's Configuration section.
``MEM0_DEFAULT_USER_ID`` makes the ``user_id`` argument optional on every
tool, which keeps single-user client configs to zero per-call arguments.

The ``mcp`` package is an optional dependency; this module is the only place
that imports it, so the core library keeps working without it.
"""

from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # pragma: no cover - exercised only without extras
    raise ImportError(
        "MCP support requires the optional dependency: "
        "pip install 'mem0-alpha[mcp]'"
    ) from e

from llm_mem0 import (
    add_memories,
    delete_memory as _delete_memory,
    format_memories_for_prompt,
    get_all_memories,
    search_memories,
    search_memories_smart,
)
from llm_mem0.settings import settings

mcp = FastMCP("mem0-alpha")


class _MissingUserId(ValueError):
    pass


def _resolve_user_id(user_id: str) -> str:
    """Explicit argument wins; fall back to MEM0_DEFAULT_USER_ID."""
    uid = (user_id or "").strip() or settings.MEM0_DEFAULT_USER_ID
    if not uid:
        raise _MissingUserId(
            "user_id is required: pass it as an argument or set the "
            "MEM0_DEFAULT_USER_ID env var in the MCP server config."
        )
    return uid


@mcp.tool()
async def add_memory(
    user_text: str,
    user_id: str = "",
    assistant_text: str = "",
    context_hint: str = "",
) -> str:
    """Store a conversation turn as long-term memory facts.

    Runs the extraction gate (one small-model call) that turns the turn into
    clean, deduplicated facts about the user before storing them.

    Args:
        user_text: What the user said.
        user_id: Stable ID of the user the facts belong to (falls back to
            the MEM0_DEFAULT_USER_ID env var).
        assistant_text: The assistant reply, for context (optional).
        context_hint: Extra context for the extractor (optional).
    """
    try:
        uid = _resolve_user_id(user_id)
    except _MissingUserId as exc:
        return f"Error: {exc}"
    result = await add_memories(
        user_text=user_text,
        assistant_text=assistant_text,
        user_id=uid,
        context_hint=context_hint,
    )
    if result is None:
        return (
            "Error: memory store unavailable (is the ChromaDB server "
            "running and reachable?)."
        )
    if isinstance(result, dict):
        if result.get("error"):
            return f"Error: {result['error']}"
        n = len(result.get("results") or [])
        if n:
            skipped = result.get("skipped") or {}
            dup = skipped.get("near_dup", 0)
            suffix = f" ({dup} duplicate(s) merged/skipped)" if dup else ""
            return f"Stored {n} fact(s).{suffix}"
        return "Processed (no new facts extracted or all were duplicates)."
    return "Stored."


@mcp.tool()
async def search_memory(
    query: str,
    user_id: str = "",
    limit: int = 5,
    smart: bool = False,
) -> str:
    """Search long-term memories; returns a prompt-ready, injection-safe block.

    Args:
        query: Natural-language search query.
        user_id: Whose memories to search (falls back to MEM0_DEFAULT_USER_ID).
        limit: Maximum number of memories to return.
        smart: Enable query-rewrite + LLM rerank (slower, higher recall).
    """
    try:
        uid = _resolve_user_id(user_id)
    except _MissingUserId as exc:
        return f"Error: {exc}"
    if smart:
        memories = await search_memories_smart(query, uid, limit=limit)
    else:
        memories = await search_memories(query, uid, limit=limit)
    if not memories:
        return "(no relevant memories)"
    return format_memories_for_prompt(memories)


@mcp.tool()
async def list_memories(
    user_id: str = "",
    limit: int = 50,
) -> str:
    """List stored memories with their IDs, for review and housekeeping.

    Returns one line per memory: ``<id>  [imp=N tier=N]  <text>``. Use the
    id with the delete_memory tool.

    Args:
        user_id: Whose memories to list (falls back to MEM0_DEFAULT_USER_ID).
        limit: Maximum number of rows to return.
    """
    try:
        uid = _resolve_user_id(user_id)
    except _MissingUserId as exc:
        return f"Error: {exc}"
    rows = await get_all_memories(uid, limit=max(int(limit), 1))
    if not rows:
        return "(no memories stored for this user)"
    lines = []
    for r in rows:
        md = r.get("metadata") or {}
        meta_bits = []
        if md.get("importance") is not None:
            meta_bits.append(f"imp={md.get('importance')}")
        if md.get("tier") is not None:
            meta_bits.append(f"tier={md.get('tier')}")
        if md.get("valid_to"):
            meta_bits.append("archived")
        meta = f" [{' '.join(meta_bits)}]" if meta_bits else ""
        text = (r.get("memory") or r.get("text") or "").replace("\n", " ")
        lines.append(f"{r.get('id', '?')}{meta}  {text[:200]}")
    return "\n".join(lines)


@mcp.tool()
async def delete_memory(memory_id: str) -> str:
    """Delete one memory by ID (see list_memories for IDs).

    Also removes the memory's rows from the side indices (entity-graph
    edges and the BM25 keyword index) so hybrid retrieval can't resurface
    the deleted fact.

    Args:
        memory_id: The memory's ID as shown by list_memories.
    """
    mid = (memory_id or "").strip()
    if not mid:
        return "Error: memory_id is required."
    ok = await _delete_memory(mid)
    if not ok:
        return f"Error: failed to delete {mid} (wrong id, or store unavailable)."
    # Side indices are best-effort mirrors; clean them so BM25/graph recall
    # can't resurface the deleted fact.
    try:
        from llm_mem0 import bm25_index, graph
        await asyncio.to_thread(graph.delete_fact_edges, mid)
        await asyncio.to_thread(bm25_index.remove_fact, mid)
    except Exception as exc:
        log.warning("side-index cleanup for %s failed (non-fatal): %s", mid, exc)
        return f"Deleted {mid} (side-index cleanup incomplete: {exc})."
    return f"Deleted {mid}."


def main() -> None:
    """Entry point for the ``mem0-alpha-mcp`` console script (stdio)."""
    # stdio transport: stdout is the protocol channel, so route our logs to
    # stderr explicitly (FastMCP does the same for its own logs).
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
