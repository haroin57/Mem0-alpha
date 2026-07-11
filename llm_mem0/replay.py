"""Sleep-like systems consolidation — Phase ③.

Two stores already exist in this library: ``history_index`` (the raw,
verbatim transcript — a hippocampus-like fast store of everything that was
said) and the curated Mem0/Chroma fact store (a slower, distilled
neocortex-like store). Without this module, the only transfer between them
happens once, at ingest time, on the turn as it was written. Human systems
consolidation instead replays hippocampal traces repeatedly (classically
during sleep) to both re-encode what the fast store already held and to
extract higher-level regularities the fast store never explicitly
represented. This module gives two batch operations that mirror that:

  1. **Replay** (``replay_channel_history``): re-run extraction over OLD raw
     history using the CURRENT entity dictionary. A proper noun mentioned
     once, early on — before it had enough later mentions to seed
     ``graph.get_top_entities`` — was invisible to the extractor at the time
     it was said. Replaying it later, once the name is "known", can recover
     the fact retroactively. Facts already known are absorbed as
     reinforcement by the existing dedup pipeline, not duplicated.

  2. **Schema abstraction** (``abstract_schemas_for_user``): human semantic
     memory holds general regularities ("likes science fiction") separately
     from the individual episodes that support them ("likes Blade Runner",
     "likes Harmony"). This clusters facts that each reference a DIFFERENT
     entity sharing a (category, subcategory, entity type), and — only when
     enough independent instances clearly support it — synthesizes a
     general-trend fact via the small model. Instances are never archived;
     the schema fact coexists as a higher-level summary.

Not the same axis as an embedding-similarity consolidation pass (which
merges near-duplicate wording regardless of age): this module recovers
facts the extractor missed the first time and generalizes across distinct
instances.

Both operations are dry-run by default and only ever ADD facts (replay's
absorbed-as-reinforcement case aside, which only touches counters) —
neither can destroy information, unlike decay.py's gist compression.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from .atomic_write import atomic_write_json
from .llm import complete_json, escape_braces
from .settings import settings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# (a) Replay
# ---------------------------------------------------------------------------
def _cursor_path() -> Path:
    return Path(settings.STATE_DIR) / "replay_cursor.json"


def _load_cursor() -> dict:
    p = _cursor_path()
    if not p.is_file():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("replay cursor load failed: %s", exc)
        return {}


def _save_cursor(data: dict) -> None:
    atomic_write_json(_cursor_path(), data)


def _pair_turns(lines: list[dict]) -> tuple[list[tuple[str, str]], str]:
    """Pair adjacent (user, assistant) lines into turns for re-extraction.

    ``channel``-role lines (ambient messages not addressed to/from the
    primary speaker) are skipped entirely — replay must not introduce
    speaker-bleed that the live ingest path already guards against. Returns
    ``(turns, last_ts_seen)`` so the caller can advance the cursor even past
    trailing lines that didn't form a complete turn.
    """
    turns: list[tuple[str, str]] = []
    last_ts = ""
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        last_ts = line["ts"] or last_ts
        if line["role"] != "user":
            i += 1
            continue
        user_text = line["text"]
        assistant_text = ""
        if i + 1 < n and lines[i + 1]["role"] == "assistant":
            assistant_text = lines[i + 1]["text"]
            last_ts = lines[i + 1]["ts"] or last_ts
            i += 2
        else:
            i += 1
        turns.append((user_text, assistant_text))
    return turns, last_ts


async def replay_channel_history(
    channel_id: int, user_id: str, *, max_turns: int | None = None, dry_run: bool = True,
) -> dict:
    """Re-run fact extraction over old history for one channel, walking
    forward from a per-channel cursor (persisted in
    ``STATE_DIR/replay_cursor.json``) so repeated nightly runs make forward
    progress through a backlog instead of reprocessing the same window.

    Returns ``{"turns_processed": int, "facts_kept": int, "cursor_advanced_to": str}``.
    In dry-run the cursor is never advanced (a failed/aborted dry-run run
    must not silently skip a window on the next real run).
    """
    from . import history_index
    from .ingest import add_memories

    if max_turns is None:
        max_turns = settings.MEM0_REPLAY_MAX_TURNS_PER_RUN

    cursor = _load_cursor()
    since_ts = cursor.get(str(channel_id), "")
    lines = await asyncio.to_thread(
        history_index.get_lines_since, channel_id, since_ts, max(max_turns * 4, 1),
    )
    turns, last_ts = _pair_turns(lines)
    turns = turns[:max_turns]

    processed = 0
    facts_kept = 0
    for user_text, assistant_text in turns:
        processed += 1
        if dry_run:
            continue
        try:
            result = await add_memories(
                user_text=user_text, assistant_text=assistant_text, user_id=user_id,
                metadata={"source": "replay", "channel_id": channel_id},
            )
        except Exception as exc:
            log.warning("replay: add_memories failed (non-fatal): %s", exc)
            continue
        if isinstance(result, dict):
            facts_kept += result.get("kept", 0) or 0

    log.info(
        "[REPLAY channel=%s user=...%s turns=%d kept=%d dry_run=%s]",
        channel_id, str(user_id)[-4:], processed, facts_kept, dry_run,
    )

    if not dry_run and last_ts and last_ts != since_ts:
        cursor[str(channel_id)] = last_ts
        _save_cursor(cursor)

    return {
        "turns_processed": processed,
        "facts_kept": facts_kept,
        "cursor_advanced_to": last_ts if not dry_run else since_ts,
    }


# ---------------------------------------------------------------------------
# (b) Schema abstraction
# ---------------------------------------------------------------------------
_SCHEMA_ABSTRACT_PROMPT = """以下は同じ話題領域に属する、ユーザー本人についての独立した具体的事実です。
これらに共通する一般的な傾向として明確に支持できる要約(スキーマ)があれば生成してください。

事実一覧:
{facts}

== ルール ==
- 個々の事実の多くが同じ一般化を明確に支持する場合のみ生成する
- 過度に一般化しない (例:「SF作品が好き」は妥当、「映画全般が好き」は過度な一般化)
- 個々の具体的事実は消えず並存するので、要約は「傾向の補足」であって「置き換え」ではない
- 支持が弱い/迷う場合は生成しない(空配列で構わない)
- 日本語1文、120字以内

以下のJSONのみ返してください(説明不要):
{{"schemas": [{{"text": "<一般化した1文>", "confidence": "<high|medium>"}}]}}"""


async def _generate_schema_facts(instance_texts: list[str]) -> list[str]:
    """Ask the small model for 0..n supported generalizations. Fail-open to
    an empty list — an abstraction opportunity missed this run is picked up
    the next time the group is still eligible; a hallucinated
    overgeneralization inserted into long-term memory is much harder to
    undo."""
    facts_block = "\n".join(
        f"- {escape_braces(t, limit=200)}" for t in instance_texts
    )
    data = await complete_json(
        system=(
            "You generalize a cluster of concrete personal facts into a "
            "clearly-supported trend statement, or emit none if "
            "unsupported. Return JSON only."
        ),
        user_message=_SCHEMA_ABSTRACT_PROMPT.format(facts=facts_block),
        model=settings.MEM0_SCHEMA_MODEL,
        max_tokens=settings.MEM0_SCHEMA_MAX_TOKENS,
        caller="replay.schema_abstract",
    )
    if not isinstance(data, dict):
        return []
    out = []
    for item in data.get("schemas", []):
        if not isinstance(item, dict):
            continue
        t = (item.get("text") or "").strip()
        if t and item.get("confidence") in ("high", "medium"):
            out.append(t)
    return out


async def abstract_schemas_for_user(
    user_id: str, *, dry_run: bool = True, max_groups: int | None = None,
) -> dict:
    """Cluster facts by (category, subcategory, entity type) — keeping at
    most one fact per DISTINCT linked entity per group, so a group only
    qualifies when it spans several different things, not several mentions
    of the same thing (that's dedup's job, not this module's) — and, for
    groups with at least ``MEM0_SCHEMA_MIN_DISTINCT_ENTITIES`` members, ask
    the model for a supported generalization.

    New schema facts are inserted through the normal ``add_memories``
    dedup/gate pipeline via its ``pre_extracted_facts`` escape hatch, so a
    schema that already exists (or nearly does) gets absorbed/reinforced
    rather than duplicated. Tagged ``source=consolidation_schema`` and
    ``derived_from=<comma-joined instance fact ids>`` for provenance.
    """
    from . import client as client_mod
    from . import graph
    from .ingest import add_memories

    if max_groups is None:
        max_groups = settings.MEM0_SCHEMA_MAX_GROUPS_PER_RUN
    min_entities = settings.MEM0_SCHEMA_MIN_DISTINCT_ENTITIES

    rows = await client_mod.get_all_memories(user_id)
    groups: dict[tuple, dict[int, dict]] = {}
    for r in rows:
        md = r.get("metadata") or {}
        if not isinstance(md, dict):
            continue
        if md.get("valid_to") or md.get("gist_of") or "schema" in (md.get("tags") or ""):
            continue
        fid = r.get("id")
        if not fid:
            continue
        entities = await asyncio.to_thread(graph.get_fact_entities, fid)
        if not entities:
            continue
        key_entity = max(entities, key=lambda e: e.mention_count)
        key = (md.get("category"), md.get("subcategory"), key_entity.type)
        by_entity = groups.setdefault(key, {})
        by_entity.setdefault(key_entity.id, r)

    result = {"groups_checked": 0, "schemas_created": 0}
    eligible = {k: v for k, v in groups.items() if len(v) >= min_entities}
    for key, by_entity in list(eligible.items())[:max_groups]:
        result["groups_checked"] += 1
        members = list(by_entity.values())
        texts = [t for t in (m.get("memory") or m.get("text") or "" for m in members) if t]
        if len(texts) < min_entities:
            continue
        schema_texts = await _generate_schema_facts(texts)
        if not schema_texts:
            continue
        member_ids = [m.get("id") for m in members if m.get("id")]
        log.info(
            "[SCHEMA_ABSTRACT user=...%s key=%s instances=%d schemas=%d dry_run=%s] %r",
            str(user_id)[-4:], key, len(members), len(schema_texts), dry_run, schema_texts,
        )
        if dry_run:
            result["schemas_created"] += len(schema_texts)
            continue

        category, subcategory, _etype = key
        for schema_text in schema_texts:
            fact = {
                "speaker": "self", "text": schema_text, "importance": 3,
                "category": category or "meta", "subcategory": subcategory or "",
                "tags": ["schema"], "attribute": "", "condition": "",
                "scope": "global", "entities": [],
            }
            try:
                res = await add_memories(
                    user_id=user_id, pre_extracted_facts=[fact],
                    metadata={
                        "source": "consolidation_schema",
                        "derived_from": ",".join(member_ids),
                    },
                )
            except Exception as exc:
                log.warning("schema insert failed (non-fatal): %s", exc)
                continue
            if isinstance(res, dict):
                result["schemas_created"] += res.get("kept", 0) or 0
    return result
