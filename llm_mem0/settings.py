"""Shared configuration defaults, resolved from environment variables.

Every setting here has a safe default so the library works out of the box;
override the env var for production deployments.
"""

from __future__ import annotations

import os


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


# Directory for local state: sqlite indices (entity graph, BM25, fact store,
# history index) and the embedded-mode Chroma database.
STATE_DIR = os.environ.get(
    "LLM_MEM0_STATE_DIR", os.path.join(os.path.expanduser("~"), ".llm_mem0", "state"),
)
os.makedirs(STATE_DIR, exist_ok=True)

# --- Model selection for this library's own helper LLM calls ---------------
MEM0_GATE_MODEL = os.environ.get("MEM0_GATE_MODEL", "claude-haiku-4-5-20251001")
MEM0_CLASSIFY_MODEL = os.environ.get("MEM0_CLASSIFY_MODEL", "claude-haiku-4-5-20251001")
MEM0_EXTRACT_MAX_TOKENS = int(os.environ.get("MEM0_EXTRACT_MAX_TOKENS", "12000"))
MEM0_EXTRACT_INPUT_CHARS = int(os.environ.get("MEM0_EXTRACT_INPUT_CHARS", "6000"))

# --- Retrieval feature flags / knobs ---------------------------------------
# Core memory: a small always-injected set of high-importance facts, separate
# from query-dependent recall.
MEM0_CORE_MEMORY_ENABLED = _bool_env("MEM0_CORE_MEMORY_ENABLED", True)
MEM0_CORE_CACHE_TTL_S = int(os.environ.get("MEM0_CORE_CACHE_TTL_S", "300"))

# Hybrid retrieval: fuse vector search with a BM25 keyword index (RRF).
MEM0_HYBRID_ENABLED = _bool_env("MEM0_HYBRID_ENABLED", True)
MEM0_BM25_RRF_K = int(os.environ.get("MEM0_BM25_RRF_K", "60"))
MEM0_BM25_LIMIT = int(os.environ.get("MEM0_BM25_LIMIT", "20"))

# HyDE: expand the query with a hypothetical answer before embedding.
MEM0_HYDE_ENABLED = _bool_env("MEM0_HYDE_ENABLED", True)

# Hide archived (superseded) rows at retrieval time.
MEM0_HIDE_ARCHIVED = _bool_env("MEM0_HIDE_ARCHIVED", True)

# Episode bundling: group facts sharing an episode id, capped for prompt size.
MEM0_EPISODE_BUNDLE_ENABLED = _bool_env("MEM0_EPISODE_BUNDLE_ENABLED", True)
MEM0_EPISODE_BUNDLE_LIMIT = int(os.environ.get("MEM0_EPISODE_BUNDLE_LIMIT", "5"))

# Scope filtering: honor per-fact scope (global/conversation/meta) at recall.
MEM0_SCOPE_FILTER_ENABLED = _bool_env("MEM0_SCOPE_FILTER_ENABLED", False)

# Content-hash dedup at ingest time (skip re-inserting identical text).
MEM0_INGEST_HASH_DEDUP = _bool_env("MEM0_INGEST_HASH_DEDUP", True)

# Fact store: an event-sourced sqlite log of attribute value changes.
MEM0_FACT_STORE_ENABLED = _bool_env("MEM0_FACT_STORE_ENABLED", False)

# History embedding: semantic (vector) recall over the raw transcript, in a
# separate Chroma collection. Off by default — each indexed line costs one
# embedding call.
MEM0_HISTORY_EMBED_ENABLED = _bool_env("MEM0_HISTORY_EMBED_ENABLED", False)
MEM0_HISTORY_EMBED_MAX_CHARS = int(os.environ.get("MEM0_HISTORY_EMBED_MAX_CHARS", "4000"))
MEM0_HISTORY_EMBED_LIMIT = int(os.environ.get("MEM0_HISTORY_EMBED_LIMIT", "5"))

# --- Ingest gate + dedup pipeline ------------------------------------------
# Extraction-gate path: run an LLM extractor, filter, then raw-insert facts.
# When disabled, ingest falls back to the memory store's built-in extractor.
MEM0_GATE_ENABLED = _bool_env("MEM0_GATE_ENABLED", True)
# Minimum importance (1-5) a fact must have to be stored by the gate path.
MEM0_GATE_MIN_IMPORTANCE = int(os.environ.get("MEM0_GATE_MIN_IMPORTANCE", "2"))

# Near-duplicate suppression at ingest time.
MEM0_CLIENT_DEDUP_ENABLED = _bool_env("MEM0_CLIENT_DEDUP_ENABLED", True)
# Legacy single-threshold dedup cutoff (distance; smaller = more similar).
# Only consulted when MEM0_DEDUP_TWO_STAGE is off.
MEM0_CLIENT_DEDUP_THRESHOLD = float(
    os.environ.get("MEM0_CLIENT_DEDUP_THRESHOLD", "0.85")
)
# Two-stage dedup: embedding neighbor search followed by an LLM judgment.
MEM0_DEDUP_TWO_STAGE = _bool_env("MEM0_DEDUP_TWO_STAGE", True)
# How many neighbors to fetch per candidate during the dedup search.
MEM0_DEDUP_SEARCH_LIMIT = int(os.environ.get("MEM0_DEDUP_SEARCH_LIMIT", "5"))
# Dry run: compute dedup decisions but never skip/merge/archive.
MEM0_DEDUP_DRY_RUN = _bool_env("MEM0_DEDUP_DRY_RUN", False)
# Detect conflicting facts (update/correction) during the two-stage check.
MEM0_CONFLICT_DETECTION_ENABLED = _bool_env(
    "MEM0_CONFLICT_DETECTION_ENABLED", True
)
# Allow merging a superset fact into the candidate instead of dropping it.
MEM0_DEDUP_SUPERSET_MERGE = _bool_env("MEM0_DEDUP_SUPERSET_MERGE", True)

# Extract entity-entity relations into the graph layer during ingest.
MEM0_RELATION_EXTRACT_ENABLED = _bool_env("MEM0_RELATION_EXTRACT_ENABLED", True)

# Structured attribute metadata (controlled vocabulary) on stored facts.
MEM0_ATTRIBUTE_ENABLED = _bool_env("MEM0_ATTRIBUTE_ENABLED", True)
# Deterministic attribute-match conflict route (independent of embeddings).
MEM0_ATTRIBUTE_CONFLICT_ENABLED = _bool_env(
    "MEM0_ATTRIBUTE_CONFLICT_ENABLED", True
)
# Store per-fact condition qualifiers (time:/context:).
MEM0_CONDITION_ENABLED = _bool_env("MEM0_CONDITION_ENABLED", True)
# Store per-fact scope (global/conversation) metadata.
MEM0_SCOPE_ENABLED = _bool_env("MEM0_SCOPE_ENABLED", True)
# Store typed values (e.g. quantities) as JSON metadata on facts.
MEM0_TYPED_VALUE_ENABLED = _bool_env("MEM0_TYPED_VALUE_ENABLED", True)
