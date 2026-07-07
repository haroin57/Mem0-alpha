"""Same flow as basic_usage.py, but driven by a standard API key instead of a
Claude Code CLI session — so it works with any provider mem0 supports.

llm_mem0 auto-detects the auth backend in this order:
  1. an existing Claude Code CLI OAuth session (~/.claude/.credentials.json)
  2. a standard API key from the environment

To force the API-key path (e.g. to use OpenAI), make sure no Claude Code CLI
credentials file is present, or point CLAUDE_CLI_CREDENTIALS_PATH at a
non-existent path, and set:

    export MEM0_LLM_PROVIDER=openai
    export OPENAI_API_KEY=sk-...
    export MEM0_LLM_MODEL=gpt-5-mini          # model for extraction/dedup/rerank
    export MEM0_EMBEDDER_PROVIDER=openai      # default; needs OPENAI_API_KEY

A ChromaDB server must be reachable (default 127.0.0.1:8765), or set
CHROMA_MODE=embedded for an on-disk store under ~/.llm_mem0/state.

Run:
    python examples/with_openai.py
"""

import asyncio
import os

# Force API-key auth for this example even if a CLI session happens to exist.
os.environ.setdefault("CLAUDE_CLI_CREDENTIALS_PATH", "/nonexistent/.credentials.json")
os.environ.setdefault("MEM0_LLM_PROVIDER", "openai")

from llm_mem0 import add_memories, format_memories_for_prompt, search_memories

USER_ID = "demo-user"


async def main() -> None:
    await add_memories(
        user_text="My favorite film is Blade Runner and I collect vinyl.",
        assistant_text="Great taste — the Vangelis score on vinyl must be something.",
        user_id=USER_ID,
    )

    memories = await search_memories("What films does the user like?", user_id=USER_ID)
    print(format_memories_for_prompt(memories))


if __name__ == "__main__":
    asyncio.run(main())
