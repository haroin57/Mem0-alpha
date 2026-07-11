"""Every env-var-backed setting, declared in one place and read dynamically.

Access pattern::

    from .settings import settings
    if settings.MEM0_HYBRID_ENABLED: ...

Each attribute access reads the environment variable at call time, so a
setting changed mid-process (tests, embedded hosts) is honored without any
module-reload tricks. The legacy import style ``from .settings import
MEM0_HYBRID_ENABLED`` keeps working via the module-level ``__getattr__``
(PEP 562) — note it freezes the value at import-site execution, exactly as
it always did.

To add a setting: add one ``_Spec`` line to ``_SPECS``. Nothing else.
"""

from __future__ import annotations

import os
from typing import Any, Callable, NamedTuple

# The small/fast model used for the library's own helper calls (extraction,
# classification, dedup judgment, rewrite, rerank...). Every *_MODEL setting
# defaults to this; override individually when one stage needs a bigger model.
DEFAULT_HELPER_MODEL = "claude-haiku-4-5-20251001"


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


class _Spec(NamedTuple):
    default: Any
    cast: Callable[[str], Any]


_SPECS: dict[str, _Spec] = {
    # --- Storage ------------------------------------------------------------
    # Directory for local state: sqlite indices (entity graph, BM25, fact
    # store, history index) and the embedded-mode Chroma database.
    "LLM_MEM0_STATE_DIR": _Spec(
        os.path.join(os.path.expanduser("~"), ".llm_mem0", "state"), str),
    # server (default): Chroma HTTP server — avoids the multi-process write
    # race that can corrupt an embedded HNSW index. embedded: on-disk under
    # STATE_DIR (single process only).
    "CHROMA_MODE": _Spec("server", str),
    "CHROMA_HOST": _Spec("127.0.0.1", str),
    "CHROMA_PORT": _Spec(8765, int),
    "MEM0_COLLECTION_NAME": _Spec("memories", str),

    # --- mem0-internal LLM + embedder ---------------------------------------
    # Model for mem0's own internal calls; empty = auth backend's default.
    "MEM0_LLM_MODEL": _Spec("", str),
    "MEM0_LLM_PROVIDER": _Spec("", str),
    "MEM0_EMBEDDER_PROVIDER": _Spec("openai", str),
    "MEM0_EMBEDDER_MODEL": _Spec("text-embedding-3-small", str),

    # --- Models / budgets for this library's own helper calls ---------------
    "MEM0_GATE_MODEL": _Spec(DEFAULT_HELPER_MODEL, str),
    "MEM0_CLASSIFY_MODEL": _Spec(DEFAULT_HELPER_MODEL, str),
    "MEM0_DEDUP_MODEL": _Spec(DEFAULT_HELPER_MODEL, str),
    "MEM0_DEDUP_MERGE_MODEL": _Spec(DEFAULT_HELPER_MODEL, str),
    "MEM0_CONFLICT_MODEL": _Spec(DEFAULT_HELPER_MODEL, str),
    "MEM0_REWRITE_MODEL": _Spec(DEFAULT_HELPER_MODEL, str),
    "MEM0_HYDE_MODEL": _Spec(DEFAULT_HELPER_MODEL, str),
    "MEM0_KEYWORD_MODEL": _Spec(DEFAULT_HELPER_MODEL, str),
    "MEM0_RERANK_MODEL": _Spec(DEFAULT_HELPER_MODEL, str),
    # Historically extract.py shadowed these with 2000/2000 while settings.py
    # declared 12000/6000 that never took effect; the effective values win.
    "MEM0_EXTRACT_MAX_TOKENS": _Spec(2000, int),
    "MEM0_EXTRACT_INPUT_CHARS": _Spec(2000, int),
    "MEM0_DEDUP_MAX_TOKENS": _Spec(1500, int),
    "MEM0_CONFLICT_MAX_TOKENS": _Spec(300, int),
    "MEM0_REWRITE_MAX_TOKENS": _Spec(500, int),
    "MEM0_HYDE_MAX_TOKENS": _Spec(200, int),
    "MEM0_KEYWORD_MAX_TOKENS": _Spec(300, int),
    "MEM0_RERANK_MAX_TOKENS": _Spec(400, int),

    # --- Retrieval feature flags / knobs -------------------------------------
    # Pre-rerank gate: drop candidates physically too far before the LLM
    # rerank sees them. score = distance (small = similar).
    "MEM0_RELEVANCE_MAX_DISTANCE": _Spec(1.2, float),
    # DEPRECATED — no longer consulted on the rerank=True path (a cosine gate
    # after an LLM rerank silently killed non-English queries). Declared so
    # env config that sets it still resolves.
    "MEM0_FINAL_RELEVANCE_MAX_DISTANCE": _Spec(0.65, float),
    # Core memory: a small always-injected set of high-importance facts,
    # separate from query-dependent recall.
    "MEM0_CORE_MEMORY_ENABLED": _Spec(True, _parse_bool),
    "MEM0_CORE_CACHE_TTL_S": _Spec(300, int),
    # Hybrid retrieval: fuse vector search with a BM25 keyword index (RRF).
    "MEM0_HYBRID_ENABLED": _Spec(True, _parse_bool),
    "MEM0_BM25_RRF_K": _Spec(60, int),
    "MEM0_BM25_LIMIT": _Spec(20, int),
    # HyDE: expand the query with a hypothetical answer before embedding.
    "MEM0_HYDE_ENABLED": _Spec(True, _parse_bool),
    # Hide archived (superseded) rows at retrieval time.
    "MEM0_HIDE_ARCHIVED": _Spec(True, _parse_bool),
    # Episode bundling: group facts sharing an episode id, capped for prompt size.
    "MEM0_EPISODE_BUNDLE_ENABLED": _Spec(True, _parse_bool),
    "MEM0_EPISODE_BUNDLE_LIMIT": _Spec(5, int),
    # Scope filtering: honor per-fact scope (global/conversation/meta) at recall.
    "MEM0_SCOPE_FILTER_ENABLED": _Spec(False, _parse_bool),
    # Reinforcement boost age-decay component.
    "MEM0_BOOST_ALPHA_AGE": _Spec(0.03, float),
    "MEM0_BOOST_AGE_HALF_LIFE_DAYS": _Spec(90.0, float),
    # Boost formula: "legacy" (hand-tuned freq+recency+age) or "actr"
    # (power-law base-level activation). legacy stays the default until the
    # ACT-R mode has been A/B compared in production.
    "MEM0_BOOST_MODE": _Spec("legacy", str),
    # ACT-R base-level activation knobs (MEM0_BOOST_MODE=actr).
    "MEM0_ACTR_DECAY_D": _Spec(0.5, float),
    "MEM0_ACTR_BOOST_ALPHA": _Spec(0.2, float),
    "MEM0_ACTR_RETRIEVAL_THRESHOLD": _Spec(-7.5, float),
    "MEM0_ACTR_NOISE_S": _Spec(1.0, float),
    # Recall-time (testing effect) reinforcement: fires when a fact is
    # surfaced by smart search; rate-limited per fact (spacing effect).
    "MEM0_RECALL_REINFORCE_ENABLED": _Spec(False, _parse_bool),
    "MEM0_RECALL_REINFORCE_MIN_INTERVAL_SEC": _Spec(6 * 3600, int),
    # Phase ④: spreading-activation graph expansion (Collins & Loftus-style
    # activation decaying with graph distance; see search._graph_expand_candidates).
    "MEM0_GRAPH_SPREAD_MAX_HOPS": _Spec(2, int),
    "MEM0_GRAPH_SPREAD_DECAY_GAMMA": _Spec(0.5, float),
    "MEM0_GRAPH_SPREAD_MIN_ACTIVATION": _Spec(0.05, float),
    # Sibling distance = BASE - WEIGHT * activation, floored at MIN. BASE
    # stays above a typical real cosine hit's distance so graph-expanded
    # facts never systematically outrank genuine vector/BM25 matches.
    "MEM0_GRAPH_SIBLING_BASE_DISTANCE": _Spec(0.9, float),
    "MEM0_GRAPH_SIBLING_ACTIVATION_WEIGHT": _Spec(0.4, float),
    "MEM0_GRAPH_SIBLING_MIN_DISTANCE": _Spec(0.3, float),
    # Session priming buffer (priming.py): flat activation top-up for
    # entities touched recently in this (user, channel).
    "MEM0_PRIMING_ACTIVATION_BONUS": _Spec(0.1, float),
    "MEM0_PRIMING_TTL_SEC": _Spec(30 * 60, int),
    "MEM0_PRIMING_MAX_ENTITIES": _Spec(20, int),
    # History embedding: semantic recall over the raw transcript, in a
    # separate Chroma collection. Off by default — each line costs one
    # embedding call.
    "MEM0_HISTORY_EMBED_ENABLED": _Spec(False, _parse_bool),
    "MEM0_HISTORY_EMBED_MAX_CHARS": _Spec(4000, int),
    "MEM0_HISTORY_EMBED_LIMIT": _Spec(5, int),

    # --- Ingest gate + dedup pipeline ----------------------------------------
    # Extraction-gate path: run an LLM extractor, filter, then raw-insert.
    # When disabled, ingest falls back to the store's built-in extractor.
    "MEM0_GATE_ENABLED": _Spec(True, _parse_bool),
    # Minimum importance (1-5) a fact must have to be stored by the gate path.
    "MEM0_GATE_MIN_IMPORTANCE": _Spec(2, int),
    # Content-hash dedup at ingest time (skip re-inserting identical text).
    "MEM0_INGEST_HASH_DEDUP": _Spec(True, _parse_bool),
    # Near-duplicate suppression at ingest time.
    "MEM0_CLIENT_DEDUP_ENABLED": _Spec(True, _parse_bool),
    # Legacy single-threshold dedup cutoff (distance; smaller = more similar).
    # Only consulted when MEM0_DEDUP_TWO_STAGE is off.
    "MEM0_CLIENT_DEDUP_THRESHOLD": _Spec(0.85, float),
    # Two-stage dedup: embedding neighbor search + an LLM judgment.
    "MEM0_DEDUP_TWO_STAGE": _Spec(True, _parse_bool),
    # How many neighbors to fetch per candidate during the dedup search.
    "MEM0_DEDUP_SEARCH_LIMIT": _Spec(5, int),
    # Dry run: compute dedup decisions but never skip/merge/archive.
    "MEM0_DEDUP_DRY_RUN": _Spec(False, _parse_bool),
    # Detect conflicting facts (update/correction) during the two-stage check.
    "MEM0_CONFLICT_DETECTION_ENABLED": _Spec(True, _parse_bool),
    # Allow merging a superset fact into the candidate instead of dropping it.
    "MEM0_DEDUP_SUPERSET_MERGE": _Spec(True, _parse_bool),
    # Extract entity-entity relations into the graph layer during ingest.
    "MEM0_RELATION_EXTRACT_ENABLED": _Spec(True, _parse_bool),
    # Structured attribute metadata (controlled vocabulary) on stored facts.
    "MEM0_ATTRIBUTE_ENABLED": _Spec(True, _parse_bool),
    # Deterministic attribute-match conflict route (independent of embeddings).
    "MEM0_ATTRIBUTE_CONFLICT_ENABLED": _Spec(True, _parse_bool),
    # Store per-fact condition qualifiers (time:/context:).
    "MEM0_CONDITION_ENABLED": _Spec(True, _parse_bool),
    # Store per-fact scope (global/conversation) metadata.
    "MEM0_SCOPE_ENABLED": _Spec(True, _parse_bool),
    # Store typed values (e.g. quantities) as JSON metadata on facts.
    "MEM0_TYPED_VALUE_ENABLED": _Spec(True, _parse_bool),
    # Fact store: an event-sourced sqlite log of attribute value changes.
    "MEM0_FACT_STORE_ENABLED": _Spec(False, _parse_bool),

    # --- Batched ingestion ----------------------------------------------------
    "MEM0_BATCH_SIZE": _Spec(10, int),
    "MEM0_BATCH_TURN_CHARS": _Spec(600, int),

    # --- MCP server -------------------------------------------------------------
    # Default user_id for MCP tools when the client doesn't pass one; keeps
    # single-user client configs (Claude Code / Desktop) to zero arguments.
    "MEM0_DEFAULT_USER_ID": _Spec("", str),

    # --- Auth bootstrap (consumed by llm_mem0.auth) ---------------------------
    "LLM_MEM0_ALLOW_TOKEN_REFRESH": _Spec(False, _parse_bool),
    "LLM_MEM0_DISABLE_KEYCHAIN": _Spec(False, _parse_bool),
    "CLAUDE_CLI_KEYCHAIN_SERVICE": _Spec("Claude Code-credentials", str),
}


class Settings:
    """Dynamic view over ``_SPECS``: every attribute read hits the env."""

    def __getattr__(self, name: str) -> Any:
        spec = _SPECS.get(name)
        if spec is None:
            raise AttributeError(f"unknown setting: {name}")
        raw = os.environ.get(name)
        if raw is None:
            return spec.default
        try:
            return spec.cast(raw)
        except (TypeError, ValueError):
            return spec.default

    @property
    def STATE_DIR(self) -> str:
        """State directory, created on first use."""
        path = self.LLM_MEM0_STATE_DIR
        os.makedirs(path, exist_ok=True)
        return path


settings = Settings()


def __getattr__(name: str) -> Any:  # PEP 562 — legacy import compatibility
    if name == "STATE_DIR":
        return settings.STATE_DIR
    if name in _SPECS:
        return getattr(settings, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
