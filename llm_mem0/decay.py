"""Forgetting — Phase ②'s counterpart to reinforce.py's strengthening.

Without this module, ``tier`` (Live/Hot/Warm/Cold) is assigned once at
ingest time and never revisited. A fact ingested as Hot stays Hot forever,
even if it is never mentioned or recalled again, and there is no path from
"rarely touched fact" to "compressed/archived" at all. ACT-R base-level
activation (reinforce.py's ``_actr_activation``) already tells us how
"available" a fact currently is — this module turns that into two
operations a caller can run periodically (e.g. a nightly batch job):

  1. **Tier decay** (``recompute_tiers_for_user``): reclassify every active
     fact's tier from its current activation. This is what lets a fact
     naturally fall out of ``search.get_core_memories`` (imp>=5 AND tier in
     {Live, Hot}) once it stops being reinforced or recalled — mirroring how
     a human's core beliefs that go unrehearsed for years drift out of
     constant working recall even though they're still "known".

  2. **Gist compression** (``gist_compress_for_user``): fuzzy-trace-theory
     -style forgetting for facts that have decayed to Cold — cluster them by
     shared entity and collapse each cluster into fewer, detail-poorer
     "gist" facts via ``dedup.synthesize_cluster``. Originals are
     soft-archived (``valid_to``), never deleted, so
     ``MEM0_HIDE_ARCHIVED=false`` always restores the verbatim originals.

Not to be confused with an embedding-similarity consolidation pass — that
kind of system merges near-duplicate WORDING regardless of age. This module
clusters Cold-tier facts by SHARED ENTITY to compress genuinely
old/rarely-touched material into a summary. The two are complementary
passes over different criteria.

Both operations are dry-run by default and are pure additions: no existing
ingest/search code path depends on either running. LLM/API token freshness
is the auth backend's concern — callers need no explicit refresh here.
"""

from __future__ import annotations

import asyncio
import logging
import time

from .settings import settings

log = logging.getLogger(__name__)

# --- Tier thresholds, expressed directly in ACT-R activation units ---------
# Activation is ln(sum of decayed presentation weights) and, with the decay
# exponent d=0.5, falls off MUCH faster than calendar intuition suggests: a
# fact mentioned exactly once decays as -0.5*ln(seconds_since), which is
# already -5.68 after just one day and -8.63 after a year. The thresholds
# (settings.MEM0_TIER_{LIVE,HOT,WARM}_ACTIVATION) are calibrated against
# that single-presentation curve (each reinforcement/recall pushes
# activation back up, which is what lets a frequently-touched fact stay
# Live/Hot far longer than a single mention would):
#   >= LIVE (-6.0)  Live  (single mention within ~a few days, or reinforced recently)
#   >= HOT  (-7.5)  Hot   (single mention within ~a month)
#   >= WARM (-8.7)  Warm  (single mention within ~a year)
#   <  WARM         Cold
# Tune via env if a corpus's actual distribution (dry-run output) skews.


def activation_to_tier(activation: float) -> int:
    """Map an ACT-R activation value to a tier: 0=Live, 1=Hot, 2=Warm, 3=Cold."""
    if activation == float("-inf"):
        return 3
    if activation >= settings.MEM0_TIER_LIVE_ACTIVATION:
        return 0
    if activation >= settings.MEM0_TIER_HOT_ACTIVATION:
        return 1
    if activation >= settings.MEM0_TIER_WARM_ACTIVATION:
        return 2
    return 3


def compute_fact_activation(metadata: dict, *, now: float | None = None) -> float:
    """Thin adapter from a fact's stored metadata to reinforce._actr_activation."""
    from .reinforce import _actr_activation
    now = now if now is not None else time.time()
    try:
        created = int(metadata.get("created_at_unix") or metadata.get("timestamp") or 0)
    except (TypeError, ValueError):
        created = 0
    try:
        mention = int(metadata.get("mention_count", 1))
    except (TypeError, ValueError):
        mention = 1
    try:
        recall = int(metadata.get("recall_count", 0))
    except (TypeError, ValueError):
        recall = 0
    reinforced_at = metadata.get("reinforced_at") or []
    return _actr_activation(created, reinforced_at, mention, recall, now=now)


def is_gist_exempt(metadata: dict) -> bool:
    """Facts that must never be silently compressed into a gist.

    - importance > MEM0_GIST_MAX_IMPORTANCE: the extractor judged this
      load-bearing; compressing away its specifics would be a real loss.
    - attribute-tagged facts: these are the current value of a
      single-valued/temporal attribute (see attribute_registry.py) and back
      the deterministic conflict route in
      ingest._attribute_conflict_check — collapsing several attribute
      history rows into a vague gist would break that lookup.
    - already a gist (``gist_of`` present): never compress a compression.
    """
    if not isinstance(metadata, dict):
        return True
    try:
        imp = int(metadata.get("importance", 0))
    except (TypeError, ValueError):
        imp = 0
    if imp > settings.MEM0_GIST_MAX_IMPORTANCE:
        return True
    if (metadata.get("attribute") or "").strip():
        return True
    if metadata.get("gist_of"):
        return True
    return False


async def recompute_tiers_for_user(user_id: str, *, dry_run: bool = True) -> dict:
    """Recompute ``tier`` for every active fact of ``user_id`` from its
    current ACT-R activation.

    ``dry_run=True`` (default) computes and logs every transition but never
    writes. Returns ``{"checked": int, "changed": int, "transitions": {...}}``
    — in dry-run, ``changed`` counts transitions that *would* be written.
    """
    from . import client as client_mod

    mem = await asyncio.to_thread(client_mod._get_mem0)
    if not mem:
        return {"checked": 0, "changed": 0, "transitions": {}}

    rows = await client_mod.get_all_memories(user_id)
    now = time.time()
    checked = 0
    changed = 0
    transitions: dict[str, int] = {}
    for r in rows:
        md = r.get("metadata") or {}
        if not isinstance(md, dict) or md.get("valid_to"):
            continue  # archived rows are frozen — decay doesn't touch them
        old_tier = md.get("tier")
        try:
            old_tier = int(old_tier)
        except (TypeError, ValueError):
            old_tier = None
        activation = compute_fact_activation(md, now=now)
        new_tier = activation_to_tier(activation)
        checked += 1
        if old_tier == new_tier:
            continue
        key = f"{old_tier}->{new_tier}"
        transitions[key] = transitions.get(key, 0) + 1
        log.info(
            "[TIER_DECAY user=...%s id=%s activation=%.3f tier=%s->%d dry_run=%s]",
            str(user_id)[-4:], r.get("id"), activation, old_tier, new_tier, dry_run,
        )
        if dry_run:
            changed += 1
            continue
        try:
            text = r.get("memory") or r.get("text") or ""
            await asyncio.to_thread(
                mem.update, memory_id=r.get("id"), data=text,
                metadata={"tier": new_tier},
            )
            changed += 1
        except Exception as exc:
            log.warning("tier decay update failed for %s (non-fatal): %s", r.get("id"), exc)
    return {"checked": checked, "changed": changed, "transitions": transitions}


async def gist_compress_for_user(
    user_id: str, *, dry_run: bool = True, max_clusters: int | None = None,
) -> dict:
    """Cluster Cold-tier, non-exempt facts by shared entity and collapse
    each cluster into fewer gist facts via ``dedup.synthesize_cluster``.

    Clustering key is the cluster member's most-mentioned linked entity
    (from the graph layer) — a fact with no linked entity is left alone
    rather than guessed into a cluster it may not belong to. Bounded to
    ``max_clusters`` (default ``MEM0_GIST_MAX_CLUSTERS_PER_RUN``)
    LLM-consolidation calls per run so a large backlog spreads across
    several nightly runs instead of one expensive burst.
    """
    from . import client as client_mod
    from . import graph
    from .dedup import synthesize_cluster

    if max_clusters is None:
        max_clusters = settings.MEM0_GIST_MAX_CLUSTERS_PER_RUN
    min_cluster_size = settings.MEM0_GIST_MIN_CLUSTER_SIZE

    mem = await asyncio.to_thread(client_mod._get_mem0)
    if not mem:
        return {"clusters": 0, "compressed": 0, "archived": 0}

    rows = await client_mod.get_all_memories(user_id)
    candidates = []
    for r in rows:
        md = r.get("metadata") or {}
        if not isinstance(md, dict) or md.get("valid_to"):
            continue
        try:
            tier = int(md.get("tier", 1))
        except (TypeError, ValueError):
            continue
        if tier != 3:
            continue
        if is_gist_exempt(md):
            continue
        candidates.append(r)

    clusters: dict[int, list[dict]] = {}
    for r in candidates:
        fid = r.get("id")
        if not fid:
            continue
        entities = await asyncio.to_thread(graph.get_fact_entities, fid)
        if not entities:
            continue  # no clustering key — left untouched, never guessed
        key_entity = max(entities, key=lambda e: e.mention_count)
        clusters.setdefault(key_entity.id, []).append(r)

    result = {"clusters": 0, "compressed": 0, "archived": 0}
    now_ts = int(time.time())
    for entity_id, members in list(clusters.items())[:max_clusters]:
        if len(members) < min_cluster_size:
            continue
        texts = [m.get("memory") or m.get("text") or "" for m in members]
        texts = [t for t in texts if t]
        if len(texts) < min_cluster_size:
            continue
        consolidated = await synthesize_cluster(texts)
        if not consolidated:
            continue  # fail-open: leave the cluster untouched
        result["clusters"] += 1
        log.info(
            "[GIST_COMPRESS user=...%s entity_id=%s members=%d->%d dry_run=%s] %r",
            str(user_id)[-4:], entity_id, len(members), len(consolidated), dry_run, consolidated,
        )
        if dry_run:
            result["compressed"] += len(consolidated)
            continue

        member_ids = [m.get("id") for m in members if m.get("id")]
        base_meta = {
            "category": (members[0].get("metadata") or {}).get("category", "meta"),
            "importance": max(1, min(
                int((m.get("metadata") or {}).get("importance", 2)) for m in members
            )),
            "tier": 3,
            "gist_of": ",".join(member_ids),
            "timestamp": now_ts,
        }
        new_ids: list[str] = []
        for gist_text in consolidated:
            try:
                res = await asyncio.to_thread(
                    mem.add, [{"role": "user", "content": gist_text}],
                    user_id=user_id, metadata=dict(base_meta), infer=False,
                )
                for rr in ((res.get("results") or []) if isinstance(res, dict) else []):
                    nid = rr.get("id")
                    if nid:
                        new_ids.append(nid)
                        await asyncio.to_thread(graph.add_edge, nid, entity_id)
            except Exception as exc:
                log.warning("gist insert failed (non-fatal): %s", exc)
        result["compressed"] += len(new_ids)

        for m in members:
            mid = m.get("id")
            if not mid:
                continue
            try:
                text = m.get("memory") or m.get("text") or ""
                await asyncio.to_thread(
                    mem.update, memory_id=mid, data=text,
                    metadata={
                        "valid_to": now_ts,
                        "gist_superseded_by": ",".join(new_ids),
                    },
                )
                result["archived"] += 1
            except Exception as exc:
                log.warning("gist archive failed for %s (non-fatal): %s", mid, exc)
    return result
