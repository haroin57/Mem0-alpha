"""Minimal end-to-end example using an existing Claude Code CLI session.

Prerequisites:
  1. You are logged into Claude Code (`~/.claude/.credentials.json` exists).
     No separate ANTHROPIC_API_KEY is needed — llm_mem0 reuses that session.
  2. A ChromaDB server is running (default: 127.0.0.1:8765):
         chroma run --host 127.0.0.1 --port 8765
     (or set CHROMA_MODE=embedded to use an on-disk store under
     ~/.llm_mem0/state instead of a server).
  3. An embedder is configured. The default embedder is OpenAI
     text-embedding-3-small, which needs OPENAI_API_KEY. To use a local
     embedder instead, set e.g. MEM0_EMBEDDER_PROVIDER=huggingface.

Run:
    python examples/basic_usage.py
"""

import asyncio

from llm_mem0 import (
    add_memories,
    format_memories_for_prompt,
    search_memories,
)

USER_ID = "demo-user"


async def main() -> None:
    # Ingest a conversation turn. The extraction gate distills clean,
    # self-attributable facts about the user from the raw turn before storing.
    await add_memories(
        user_text="I just moved to Berlin and started learning Rust.",
        assistant_text="Nice! How are you finding the borrow checker?",
        user_id=USER_ID,
    )

    # Later — retrieve facts relevant to a new query.
    memories = await search_memories("Where does the user live?", user_id=USER_ID)

    # Format them as a prompt-injection-safe context block to prepend to your
    # system/user prompt.
    print(format_memories_for_prompt(memories))


if __name__ == "__main__":
    asyncio.run(main())
