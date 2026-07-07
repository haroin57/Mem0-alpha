# Mem0α

English | [日本語](README.ja.md)

Long-term memory for LLM agents, provider-agnostic. Mem0α is a thin layer on
top of [`mem0`](https://github.com/mem0ai/mem0) that adds the retrieval and
ingestion polish a conversational agent actually needs in practice. Its auth
layer is pluggable and can **reuse the Claude Code CLI session you already
have**, so most of the time you don't even need an API key.

> Installed as `mem0-alpha`, imported as `llm_mem0`.

```python
from llm_mem0 import add_memories, search_memories, format_memories_for_prompt

# Pull just the facts about the user out of one conversation turn, and store them.
await add_memories(
    user_text="I moved to Berlin and picked up Rust.",
    assistant_text="Nice — how's the borrow checker treating you?",
    user_id="alice",
)

# Later, fetch what's relevant to a different query, ready to drop into a prompt.
memories = await search_memories("Where does the user live?", user_id="alice")
print(format_memories_for_prompt(memories))
# [Long-term memory — facts about this user]
# - The user lives in Berlin [imp=4 Hot] [2026-07-08]
# - The user is learning Rust [imp=3 Hot] [2026-07-08]
# [End of Long-term memory]
```

## How this differs from plain `mem0`

`mem0` handles the vector-store plumbing for you. Mem0α layers on the things
you end up needing when an agent has to remember someone over the long haul.

- **An extraction gate.** One small-model call turns a raw turn into clean
  facts *about the user* — no bleed-through from the assistant's replies — and
  validates them before anything is stored.
- **Two-stage dedup.** A cosine pass narrows the candidates, then an LLM
  decides whether to merge. A controlled attribute vocabulary means a new value
  (weight, address, job…) actually replaces the old one, instead of the same
  fact piling up every time it's reworded.
- **Hybrid retrieval.** Vector search and a BM25 keyword index are fused with
  RRF, optionally widened by query rewriting and HyDE, then reranked — so exact
  proper nouns (names, track titles) don't get lost in embedding space.
- **An entity/relation graph** with aliases, co-occurrence, and one-hop
  expansion.
- **Injection-safe formatting.** Retrieved text is stripped of markers that
  could impersonate the prompt scaffold before it goes back into a prompt.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Auth — reuse your CLI login, or bring an API key

`llm_mem0` picks an auth backend automatically, in this order:

1. **Your Claude Code CLI session.** It borrows the OAuth session the `claude`
   command already keeps, so **no separate `ANTHROPIC_API_KEY` is needed**. It
   reads the token, adds the required `anthropic-beta: oauth-2025-04-20` header,
   and patches the Anthropic SDK to use Bearer auth. Where the token lives
   depends on the OS: **Linux and Windows** keep it in
   `~/.claude/.credentials.json` (or under `$CLAUDE_CONFIG_DIR`), while
   **macOS** stores it in the login **Keychain**, so the library reads it from
   there (falling back to the file). This just reuses *your own* login; it does
   not spoof the CLI's network fingerprint.

   **Token refresh is handled conservatively, because the top priority is never
   breaking your CLI login.** When the token has expired, the library first
   re-reads the CLI's own store (Keychain/file) — if the CLI is running it keeps
   the token fresh, and Mem0α just rides along without rotating anything. Only
   if it's *still* expired does behavior diverge by source. A **file**-sourced
   token is refreshed and written back to the same file the CLI reads, so the
   two stay in sync. A **Keychain**-sourced token is not refreshed by default:
   refreshing rotates it, and there's no reliable way to write the new token
   back to the Keychain, so doing it would invalidate the CLI's own login.
   Instead the library logs a note that running any `claude` command will
   refresh the token, and returns no memory for that call. Set
   `LLM_MEM0_ALLOW_TOKEN_REFRESH=1` to refresh directly anyway (accepting that
   you may have to log back into `claude`).
2. **A standard API key.** If no CLI session is found, it falls back to an API
   key from the environment, which works with any provider `mem0` supports.

```bash
# Case 1: already logged into Claude Code? Nothing to set — it just works.

# Case 2: use OpenAI (or any provider) via an API key
export MEM0_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export MEM0_LLM_MODEL=gpt-5-mini
```

> Codex CLI's local credential cache (`~/.codex/auth.json`) is deliberately not
> reused — OpenAI documents it as an internal implementation detail with no
> supported way for third-party tools to read it. Use OpenAI/Codex models with a
> standard `OPENAI_API_KEY` instead.

## Install

```bash
pip install llm-mem0
# with the Anthropic SDK (for CLI reuse or an Anthropic API key):
pip install "llm-mem0[anthropic]"
# with the OpenAI SDK:
pip install "llm-mem0[openai]"
# with the Japanese tokenizer for the BM25 index (optional):
pip install "llm-mem0[japanese]"
```

## Before you run it

- **A vector store.** By default `llm_mem0` talks to a ChromaDB HTTP server:
  ```bash
  chroma run --host 127.0.0.1 --port 8765
  ```
  If you'd rather not run a server, set `CHROMA_MODE=embedded` to keep the data
  on disk under `~/.llm_mem0/state` (single process only).
- **An embedder.** The default is OpenAI `text-embedding-3-small`, which needs
  `OPENAI_API_KEY`. Swap it out with `MEM0_EMBEDDER_PROVIDER` /
  `MEM0_EMBEDDER_MODEL` (a local HuggingFace model, for instance).

## Public API

| Function | What it does |
|---|---|
| `add_memories(user_text=, assistant_text=, user_id=, …)` | Extract the user's facts from a turn and store them. |
| `search_memories(query, user_id, limit=8)` | Semantic search with a relevance gate. |
| `search_memories_multi(queries, user_id, …)` | Search across several queries at once. |
| `search_memories_smart(query, user_id, rewrite=True, rerank=True)` | Search with query rewriting, HyDE, and reranking. |
| `should_use_memory_llm_mode(...)` | Heuristic for whether the smart path is worth the cost. |
| `format_memories_for_prompt(memories)` | Render memories into an injection-safe prompt block. |
| `format_history_for_prompt(hits)` | Render history-search hits into a prompt block. |
| `extract_facts_for_self(user_text, assistant_text, …)` | Just the extraction step (returns facts only). |
| `get_all_memories(user_id, limit=None)` / `delete_memory(memory_id)` | Low-level housekeeping. |

## Configuration

Everything is an environment variable with a safe default (all defined in
`llm_mem0/settings.py`). The ones you'll reach for most:

| Env var | Default | Meaning |
|---|---|---|
| `CHROMA_MODE` | `server` | `server` or `embedded` |
| `CHROMA_HOST` / `CHROMA_PORT` | `127.0.0.1` / `8765` | Chroma server address |
| `MEM0_COLLECTION_NAME` | `memories` | Chroma collection name |
| `MEM0_LLM_PROVIDER` | auto | `anthropic` / `openai` / … (when using an API key) |
| `MEM0_LLM_MODEL` | backend default | model for extraction, dedup, and reranking |
| `MEM0_EMBEDDER_PROVIDER` / `MEM0_EMBEDDER_MODEL` | `openai` / `text-embedding-3-small` | the embedder |
| `LLM_MEM0_STATE_DIR` | `~/.llm_mem0/state` | sqlite indices and embedded Chroma |
| `MEM0_HYBRID_ENABLED` | `true` | fuse BM25 with vector search |
| `MEM0_HYDE_ENABLED` | `true` | widen the query with a hypothetical answer (HyDE) |

## Examples

- [`examples/basic_usage.py`](examples/basic_usage.py) — reuse a Claude Code CLI session.
- [`examples/with_openai.py`](examples/with_openai.py) — drive it with an OpenAI API key.

## License

MIT — see [LICENSE](LICENSE).
