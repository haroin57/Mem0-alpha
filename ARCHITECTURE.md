# Architecture

English | [日本語](ARCHITECTURE.ja.md)

`llm_mem0` is a thin layer on top of [`mem0`](https://github.com/mem0ai/mem0).
`mem0` already handles the hard vector-store work — embedding, upsert,
similarity search — and is multi-provider. Mem0α adds the retrieval and
ingestion polish a conversational agent needs on top of that raw store, plus a
pluggable auth layer so you don't have to provision a separate API key when
you're already logged into an LLM CLI.

## Why not use mem0 as-is

`mem0`'s built-in extractor emits terse English like `Prefers ~`, and it mixes
speakers when a turn contains both the user's and the assistant's words. For an
agent that has to remember one person over time, that led to three problems
Mem0α fixes:

1. **Speaker bleed.** Facts about the assistant or other people leaked into the
   user's memory. Here, extraction is limited to facts attributable to the user
   (`speaker == "self"`) and validated before storage.
2. **Dedup bloat.** The extractor reworded the same fact every turn, so
   near-duplicates stacked up. Here, two-stage dedup (a cosine pass to narrow
   candidates, then an LLM merge decision) collapses them, and an attribute
   registry replaces single-valued attributes (weight, address…) old value and
   all.
3. **Missed recall.** Plain vector search couldn't find exact proper nouns
   (names, track titles) that embed next to unrelated neighbors. Here, retrieval
   is hybrid: vector search and a BM25 keyword index fused with Reciprocal Rank
   Fusion, optionally widened with query rewriting and HyDE, then reranked.

## Module map

| Module | Responsibility |
|---|---|
| `auth/` | Pluggable auth backends (detailed below). |
| `client.py` | The `mem0.Memory` singleton and config builder; low-level `get_all_memories` / `delete_memory`. |
| `settings.py` | Every env-var-backed default, in one place. |
| `ingest.py` | `add_memories` — the ingestion entry point (extraction gate, hash dedup, near-duplicate merge, graph/BM25/fact-store indexing). |
| `extract.py` | Small-model extraction of the user's facts and entity relations from a turn; classification (category, importance, tier). |
| `dedup.py` | Two-stage dedup: cosine candidate gate, then an LLM merge/conflict decision. |
| `conflict.py` | Decides whether a new fact updates, contradicts, or coexists with an existing one. |
| `attribute_registry.py` | Controlled vocabulary for single-valued attributes, so a new value cleanly retires the old one. |
| `reinforce.py` | Reinforcement for repeated facts — read-modify-write of mention counts and last-seen. |
| `search.py` | `search_memories` / `search_memories_multi` / `search_memories_smart` — hybrid retrieval, query rewriting, HyDE, reranking, graph expansion. |
| `graph.py` | A SQLite entity/relation graph (aliases, co-occurrence, one-hop expansion). |
| `bm25_index.py` | A SQLite FTS5 keyword index over facts (CJK pre-tokenized via `morpho`). |
| `history_index.py` / `history_embed.py` | Keyword/embedding indices over the full conversation history, kept separate from the curated fact store. |
| `morpho.py` | Optional Japanese morphological tokenization for the FTS index; falls back cleanly if the tokenizer isn't installed. |
| `fact_store.py` | An optional event-sourced SQLite log of attribute value changes. |
| `format.py` | Renders retrieved memories and history hits into a prompt block. |
| `sanitize.py` | Injection defense for untrusted text (stripping prompt-scaffold and power-grab markers). |
| `batch.py` | A per-conversation turn accumulator for batched ingestion. |

## The auth layer

The one genuinely novel piece. LLM calls happen in two places, and the auth
backend serves both from a single set of credentials:

1. **mem0's own internal calls** — the extraction and embedding it runs inside
   `Memory.add()` and `search()`. The backend supplies `mem0_llm_config()`, the
   `{"provider", "config"}` block `Memory.from_config()` reads.
2. **This library's own helper calls** — self-fact extraction, metadata
   classification, dedup decisions, query rewriting and reranking. The backend
   supplies `complete(system, user_message, model, max_tokens) -> str`, a
   provider-agnostic single-shot completion that returns `""` on failure, so
   callers never have to special-case an SDK.

### Backends (auto-detected, in order)

- **`AnthropicCliAuth`** reuses an existing **Claude Code CLI** OAuth session.
  Where the credentials live depends on the OS: on Linux it reads
  `~/.claude/.credentials.json` (the same file `claude` reads and refreshes),
  and on macOS the CLI keeps them in the login **Keychain**, so the backend
  reads them via `security find-generic-password`, falling back to the file. It
  adds the required `anthropic-beta: oauth-2025-04-20` header (an OAuth token is
  rejected 401 without it) and patches `anthropic.Anthropic.__init__` so the
  client mem0 builds internally uses Bearer auth instead of `x-api-key` —
  guarded by thread-safe double-checked locking. **This reuses your own login;
  it does not spoof the CLI's network fingerprint** — requests go out through
  the stock Anthropic SDK.

  **Token freshness and CLI coexistence.** When the cached token has expired,
  the backend first re-reads the CLI's own store (Keychain/file): if the CLI is
  running it keeps that token fresh, so Mem0α rides along without rotating
  anything. Only if the token is still expired does it refresh directly — and it
  logs a warning, because a direct refresh rotates the refresh token and
  invalidates the copy the CLI holds, which can force a `claude` re-login. That
  ordering is what keeps the library from silently breaking your CLI login.
- **`ApiKeyAuth`** uses a standard API key from the environment
  (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …). For mem0's internal calls this
  path works with any provider mem0 supports; `complete()` is implemented
  natively for Anthropic and OpenAI.

Selection lives in `auth/__init__.py:get_auth_backend()`, a process-wide cached
singleton. To add a provider, implement `AuthBackend` (`auth/base.py`) and
register it there.

> **A note on scope.** Codex CLI's local credential cache (`~/.codex/auth.json`)
> is deliberately not reused: OpenAI documents it as an internal implementation
> detail with no supported third-party read path, and its schema is unpublished.
> Use OpenAI/Codex models with a standard `OPENAI_API_KEY` instead.

## Storage

- **Vectors** live in ChromaDB — either an HTTP server (`CHROMA_MODE=server`,
  the default, which avoids the multi-process write race that can corrupt an
  embedded HNSW index) or an embedded on-disk store (`CHROMA_MODE=embedded`).
- **Everything else** — the entity graph, BM25 index, history index, and fact
  store — lives in SQLite files under `LLM_MEM0_STATE_DIR` (default
  `~/.llm_mem0/state`), each in its own file so any one index can be rebuilt or
  wiped independently.

## Configuration

Every knob is an environment variable with a safe default, all gathered in
`settings.py`. The main ones are `CHROMA_MODE` / `CHROMA_HOST` / `CHROMA_PORT`,
`MEM0_COLLECTION_NAME`, `MEM0_LLM_PROVIDER` / `MEM0_LLM_MODEL`,
`MEM0_EMBEDDER_PROVIDER` / `MEM0_EMBEDDER_MODEL`, and the retrieval feature flags
(`MEM0_HYBRID_ENABLED`, `MEM0_HYDE_ENABLED`, `MEM0_SCOPE_FILTER_ENABLED`, …).
