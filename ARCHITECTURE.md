# Architecture

`llm_mem0` is a thin, opinionated layer over [`mem0`](https://github.com/mem0ai/mem0).
`mem0` already handles the hard vector-store plumbing (embedding, upsert,
similarity search) and is multi-provider. This library adds the retrieval-
and ingestion-quality machinery that a conversational agent needs on top of a
raw vector store, plus a pluggable auth layer so you don't have to provision a
separate API key when you already have an LLM CLI logged in.

## Why not just use mem0 directly?

`mem0`'s built-in extractor produces terse English `Prefers ~` statements and
mixes speakers when a turn contains both user and assistant text. For a
long-running personal agent that caused three problems this library fixes:

1. **Speaker bleed** — facts about the assistant/other participants leaked
   into the user's memory. Here, extraction is gated to *self-attributable*
   facts (`speaker == "self"`) and validated before insert.
2. **Dedup bloat** — near-duplicate facts accumulated because the extractor
   reworded the same fact each turn. Here, a two-stage dedup (cosine
   pre-filter → LLM merge judgment) collapses them, and an attribute registry
   deterministically supersedes single-valued attributes (weight, residence…).
3. **Recall misses** — pure vector search missed exact proper nouns (names,
   track titles) that embed near unrelated neighbors. Here, retrieval is
   hybrid: vector search fused with a BM25 keyword index via Reciprocal Rank
   Fusion, optionally expanded with query rewrite + HyDE and reranked.

## Module map

| Module | Responsibility |
|---|---|
| `auth/` | Pluggable authentication backends (see below). |
| `client.py` | The `mem0.Memory` singleton + config builder; low-level `get_all_memories` / `delete_memory`. |
| `settings.py` | All env-var-backed configuration defaults in one place. |
| `ingest.py` | `add_memories` — the ingestion entry point (extraction gate, hash dedup, near-dup merge, graph/BM25/fact-store indexing). |
| `extract.py` | Small-model extraction of self-facts + entity relations from a turn; taxonomy/importance/tier classification. |
| `dedup.py` | Two-stage dedup: cosine candidate gate + LLM merge/conflict judgment. |
| `conflict.py` | Classifies whether a new fact updates, contradicts, or coexists with an existing one. |
| `attribute_registry.py` | Controlled vocabulary for single-valued attributes so a new value deterministically archives the old. |
| `reinforce.py` | Read-modify-write reinforcement (mention counts, last-seen) for repeated facts. |
| `search.py` | `search_memories` / `search_memories_multi` / `search_memories_smart` — hybrid retrieval, query rewrite, HyDE, rerank, graph expansion. |
| `graph.py` | SQLite entity/relation graph (aliases, co-occurrence, 1-hop expansion). |
| `bm25_index.py` | SQLite FTS5 keyword index over facts (CJK pre-tokenized via `morpho`). |
| `history_index.py` / `history_embed.py` | Full-conversation-history keyword/embedding index, separate from the curated fact store. |
| `morpho.py` | Optional Japanese morphological tokenization for the FTS index (falls back gracefully when the tokenizer isn't installed). |
| `fact_store.py` | Optional event-sourced SQLite log of attribute value changes. |
| `format.py` | Renders retrieved memories / history hits into a prompt block. |
| `sanitize.py` | Prompt-injection sentinel/power-phrase scrubbing for untrusted retrieved text. |
| `batch.py` | Per-conversation turn accumulator for batched ingestion. |

## Authentication layer

The one genuinely novel piece. LLM calls happen in two places, and the auth
backend serves both from a single credential source:

1. **mem0's own internal calls** (its extractor/embedder during `Memory.add()`
   and `search()`) — the backend supplies `mem0_llm_config()`, the
   `{"provider", "config"}` block `Memory.from_config()` consumes.
2. **This library's direct helper calls** (self-fact extraction, metadata
   classification, dedup judgment, query rewrite/rerank) — the backend
   supplies `complete(system, user_message, model, max_tokens) -> str`, a
   provider-agnostic single-turn completion that returns `""` on failure so
   callers never have to special-case an SDK.

### Backends (auto-detected in order)

- **`AnthropicCliAuth`** — reuses an existing **Claude Code CLI** OAuth session.
  Credential source is platform-specific: on Linux it reads
  `~/.claude/.credentials.json` (the same file the `claude` CLI reads and
  refreshes); on macOS the CLI keeps credentials in the login **Keychain**, so
  the backend reads them via `security find-generic-password` and falls back to
  the file. It sends the mandatory `anthropic-beta: oauth-2025-04-20` header
  (an OAuth token without it is rejected 401) and monkey-patches
  `anthropic.Anthropic.__init__` so mem0's internally-constructed client uses
  Bearer auth instead of `x-api-key` — guarded by thread-safe double-checked
  locking. **This reuses your own authenticated session; it does not spoof the
  CLI's network fingerprint** — requests go out through the stock Anthropic SDK.

  **Token freshness / CLI co-existence.** When the cached token is expired the
  backend first *re-reads* the CLI's own store (Keychain/file): if the CLI is
  running it keeps that token fresh, so we piggyback on its refresh cycle
  without rotating anything. Only if the token is still expired does the backend
  refresh directly — and it logs a warning, because a direct refresh rotates the
  refresh token and invalidates the copy the CLI holds, which can force a
  `claude` re-login. This ordering keeps the library from silently breaking the
  user's CLI login.
- **`ApiKeyAuth`** — standard API key from the environment (`ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`, …). This is the path that makes the library work with any
  provider mem0 supports for its internal calls; `complete()` is implemented
  natively for Anthropic and OpenAI.

Selection lives in `auth/__init__.py:get_auth_backend()` (a process-wide cached
singleton). To add a provider, implement `AuthBackend` (`auth/base.py`) and
register it there.

> **Scope note.** Codex CLI's local credential cache (`~/.codex/auth.json`) is
> intentionally **not** reused: OpenAI documents it as an internal
> implementation detail with no supported third-party read path, and its schema
> is unpublished. Use OpenAI/Codex models through a standard `OPENAI_API_KEY`
> instead.

## Storage

- **Vectors**: ChromaDB, either an HTTP server (`CHROMA_MODE=server`, default —
  avoids the multi-process write race that can corrupt an embedded HNSW index)
  or an embedded on-disk store (`CHROMA_MODE=embedded`).
- **Everything else** (entity graph, BM25 index, history index, fact store):
  SQLite files under `LLM_MEM0_STATE_DIR` (default `~/.llm_mem0/state`), each in
  its own file so indices can be rebuilt or wiped independently.

## Configuration

Every knob is an environment variable with a safe default, centralized in
`settings.py`. Key ones: `CHROMA_MODE` / `CHROMA_HOST` / `CHROMA_PORT`,
`MEM0_COLLECTION_NAME`, `MEM0_LLM_PROVIDER` / `MEM0_LLM_MODEL`,
`MEM0_EMBEDDER_PROVIDER` / `MEM0_EMBEDDER_MODEL`, and the retrieval feature
flags (`MEM0_HYBRID_ENABLED`, `MEM0_HYDE_ENABLED`, `MEM0_SCOPE_FILTER_ENABLED`,
…).
