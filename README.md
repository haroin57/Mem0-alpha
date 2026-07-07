# llm-mem0

Provider-agnostic **long-term memory for LLM agents** — a thin, opinionated
layer over [`mem0`](https://github.com/mem0ai/mem0) that adds the retrieval-
and ingestion-quality machinery a conversational agent actually needs, plus a
pluggable auth layer that can **reuse your existing Claude Code CLI session** so
you don't have to provision a separate API key.

```python
from llm_mem0 import add_memories, search_memories, format_memories_for_prompt

# Distill clean, self-attributable facts from a conversation turn and store them.
await add_memories(
    user_text="I just moved to Berlin and started learning Rust.",
    assistant_text="Nice! How's the borrow checker treating you?",
    user_id="alice",
)

# Later — retrieve what's relevant to a new query, ready to drop into a prompt.
memories = await search_memories("Where does the user live?", user_id="alice")
print(format_memories_for_prompt(memories))
# [Long-term memory — facts about this user]
# - The user lives in Berlin [imp=4 Hot] [2026-07-08]
# - The user is learning Rust [imp=3 Hot] [2026-07-08]
# [End of Long-term memory]
```

## Why not just use `mem0` directly?

`mem0` gives you the vector-store plumbing. This library adds what a
long-running personal agent needs on top of it:

- **Self-fact extraction gate** — one small-model call turns a raw turn into
  clean facts *about the user only* (no speaker bleed from assistant text),
  validated before insert.
- **Two-stage dedup** — cosine pre-filter → LLM merge judgment, plus a
  controlled attribute vocabulary so a new value (weight, residence, job…)
  deterministically supersedes the old one instead of piling up.
- **Hybrid retrieval** — vector search fused with a BM25 keyword index (RRF),
  optionally expanded with query rewrite + HyDE and reranked, so exact proper
  nouns (names, track titles) aren't lost in embedding space.
- **Entity/relation graph** — aliases, co-occurrence, and 1-hop expansion.
- **Injection-safe formatting** — retrieved text is scrubbed of prompt-scaffold
  markers before it re-enters a prompt.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Authentication — bring your CLI session or any API key

`llm_mem0` auto-detects an auth backend in this order:

1. **Claude Code CLI session** — reuses the OAuth session the `claude` CLI
   already maintains, so **no separate `ANTHROPIC_API_KEY` is needed**. It reads
   the token, sends the required `anthropic-beta: oauth-2025-04-20` header, and
   patches the Anthropic SDK to use Bearer auth. On **Linux** the token comes
   from `~/.claude/.credentials.json`; on **macOS** the CLI stores it in the
   login **Keychain**, so the library reads it from there (falling back to the
   file). This reuses *your own* authenticated session — it does not spoof the
   CLI's network fingerprint. When a token is expired the library re-reads the
   CLI's own store first and only refreshes directly as a last resort (a direct
   refresh rotates the token and can force a `claude` re-login).
2. **Standard API key** — otherwise it falls back to an API key from the
   environment, which works with any provider `mem0` supports.

```bash
# Path 1: already logged into Claude Code? Nothing to set — it just works.

# Path 2: use OpenAI (or any provider) via an API key
export MEM0_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export MEM0_LLM_MODEL=gpt-5-mini
```

> Codex CLI's local credential cache (`~/.codex/auth.json`) is intentionally
> **not** reused — OpenAI documents it as an internal implementation detail
> with no supported third-party read path. Use OpenAI/Codex models through a
> standard `OPENAI_API_KEY`.

## Install

```bash
pip install llm-mem0
# with the Anthropic SDK (for Claude Code CLI reuse or Anthropic API keys):
pip install "llm-mem0[anthropic]"
# with the OpenAI SDK:
pip install "llm-mem0[openai]"
# optional Japanese tokenization for the BM25 index:
pip install "llm-mem0[japanese]"
```

## Prerequisites

- **A vector store.** By default `llm_mem0` talks to a ChromaDB HTTP server:
  ```bash
  chroma run --host 127.0.0.1 --port 8765
  ```
  Or set `CHROMA_MODE=embedded` to use an on-disk store under
  `~/.llm_mem0/state` (single process only).
- **An embedder.** The default is OpenAI `text-embedding-3-small`
  (`OPENAI_API_KEY` required). Override with `MEM0_EMBEDDER_PROVIDER` /
  `MEM0_EMBEDDER_MODEL` (e.g. a local HuggingFace embedder).

## Public API

| Function | Purpose |
|---|---|
| `add_memories(user_text=, assistant_text=, user_id=, …)` | Extract self-facts from a turn and store them. |
| `search_memories(query, user_id, limit=8)` | Semantic retrieval with a relevance-distance gate. |
| `search_memories_multi(queries, user_id, …)` | Fan retrieval over several queries. |
| `search_memories_smart(query, user_id, rewrite=True, rerank=True)` | Query-rewrite + HyDE + rerank retrieval. |
| `should_use_memory_llm_mode(...)` | Heuristic gate for whether to spend the smart path. |
| `format_memories_for_prompt(memories)` | Render memories as an injection-safe prompt block. |
| `format_history_for_prompt(hits)` | Render history-search hits as a prompt block. |
| `extract_facts_for_self(user_text, assistant_text, …)` | Just the extraction step (facts only). |
| `get_all_memories(user_id, limit=None)` / `delete_memory(memory_id)` | Low-level housekeeping. |

## Configuration

Every setting is an environment variable with a safe default (all defined in
`llm_mem0/settings.py`). The common ones:

| Env var | Default | Meaning |
|---|---|---|
| `CHROMA_MODE` | `server` | `server` or `embedded` |
| `CHROMA_HOST` / `CHROMA_PORT` | `127.0.0.1` / `8765` | Chroma server address |
| `MEM0_COLLECTION_NAME` | `memories` | Chroma collection |
| `MEM0_LLM_PROVIDER` | auto | `anthropic` / `openai` / … (for API-key mode) |
| `MEM0_LLM_MODEL` | backend default | model for extraction/dedup/rerank |
| `MEM0_EMBEDDER_PROVIDER` / `MEM0_EMBEDDER_MODEL` | `openai` / `text-embedding-3-small` | embedder |
| `LLM_MEM0_STATE_DIR` | `~/.llm_mem0/state` | sqlite indices + embedded Chroma |
| `MEM0_HYBRID_ENABLED` | `true` | fuse BM25 with vector search |
| `MEM0_HYDE_ENABLED` | `true` | hypothetical-answer query expansion |

## Examples

- [`examples/basic_usage.py`](examples/basic_usage.py) — reuse a Claude Code CLI session.
- [`examples/with_openai.py`](examples/with_openai.py) — drive it with an OpenAI API key.

## License

MIT — see [LICENSE](LICENSE).
