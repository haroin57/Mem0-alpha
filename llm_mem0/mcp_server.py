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
        "mem0-alpha": {"command": "mem0-alpha-mcp"}
      }
    }

Configuration (vector store path, models, auth backend) is inherited from
``llm_mem0.settings`` env vars — see the README's Configuration section.

The ``mcp`` package is an optional dependency; this module is the only place
that imports it, so the core library keeps working without it.
"""

from __future__ import annotations

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
    format_memories_for_prompt,
    search_memories,
    search_memories_smart,
)

mcp = FastMCP("mem0-alpha")


@mcp.tool()
async def add_memory(
    user_text: str,
    user_id: str,
    assistant_text: str = "",
    context_hint: str = "",
) -> str:
    """Store a conversation turn as long-term memory facts.

    Runs the extraction gate (one small-model call) that turns the turn into
    clean, deduplicated facts about the user before storing them.

    Args:
        user_text: What the user said.
        user_id: Stable ID of the user the facts belong to.
        assistant_text: The assistant reply, for context (optional).
        context_hint: Extra context for the extractor (optional).
    """
    result = await add_memories(
        user_text=user_text,
        assistant_text=assistant_text,
        user_id=user_id,
        context_hint=context_hint,
    )
    if isinstance(result, dict):
        n = len(result.get("results") or [])
        return f"Stored {n} fact(s)." if n else "Processed (no new facts)."
    if result:
        return "Stored."
    return "No new facts extracted (nothing stored)."


@mcp.tool()
async def search_memory(
    query: str,
    user_id: str,
    limit: int = 5,
    smart: bool = False,
) -> str:
    """Search long-term memories; returns a prompt-ready, injection-safe block.

    Args:
        query: Natural-language search query.
        user_id: Whose memories to search.
        limit: Maximum number of memories to return.
        smart: Enable query-rewrite + LLM rerank (slower, higher recall).
    """
    if smart:
        memories = await search_memories_smart(query, user_id, limit=limit)
    else:
        memories = await search_memories(query, user_id, limit=limit)
    if not memories:
        return "(no relevant memories)"
    return format_memories_for_prompt(memories)


def main() -> None:
    """Entry point for the ``mem0-alpha-mcp`` console script (stdio)."""
    mcp.run()


if __name__ == "__main__":
    main()
