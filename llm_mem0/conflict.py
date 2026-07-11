"""Contradiction classification for Mem0 fact ingestion (Phase B-2).

When dedup says "novel but close to an existing fact" we still want to
know *how* the new fact relates to the old one:

    update      — new fact replaces the old (state change / time progression)
                  e.g. "髪は黒" → "髪はハイトーン"
    correction  — old fact was wrong; the new one is the right version
                  e.g. "Alice に任せる" → "やっぱり Bob"
    addition    — different aspects of the same entity, both should stay
                  e.g. "Rust が好き" + "TypeScript が好き"
    none        — no real conflict (probably a pure paraphrase dedup
                  already handled)

The ingest pipeline turns ``update`` and ``correction`` into a soft
archive: the old fact gets metadata ``valid_to=now`` and a tag of which
conflict type retired it. Retrieval hides facts with ``valid_to <= now``
unless ``MEM0_HIDE_ARCHIVED=false``, so the rollback path is "flip the
env var, restart, all archived facts reappear."

Fail-open contract: when the LLM call fails or the response is
unparseable, classification returns ``addition`` so a hiccup never
costs us a real fact.
"""

from __future__ import annotations

import logging

from .llm import complete_json, escape_braces
from .settings import settings

log = logging.getLogger(__name__)


_CONFLICT_PROMPT = """以下の2つの fact が矛盾するか、または片方がもう片方を時間的に上書きするかを判定してください。

既存: {existing}
新規: {candidate}

判定:
- "update": 新規が既存を時間的に上書き (例: 髪色変更、住所変更、状態変化)
- "correction": 既存が誤りで新規が訂正 (例: 「Alice に任せる」→「やっぱ Bob」)
- "addition": 別 aspect の事実で両方残すべき (例: Rust好き + TypeScript好き)
- "none": 矛盾でも上書きでもない (同一表現の言い換え等 / dedup でハンドル済み想定)

迷ったら "addition" にしてください (false positive で消すより残す方が安全)。

以下のJSONのみ返してください(説明不要):
{{"conflict_type":"<update|correction|addition|none>","reason":"<30字以内>"}}"""


VALID_CONFLICT_TYPES = ("update", "correction", "addition", "none")


async def classify_conflict(existing_text: str, candidate_text: str) -> dict:
    """Ask the helper model how the candidate relates to the existing fact.

    Returns ``{"conflict_type": <one of VALID_CONFLICT_TYPES>, "reason": str}``.
    On any failure the result is ``{"conflict_type": "addition", "reason":
    "fail-open"}`` — i.e. keep both facts and let downstream dedup / rerank
    handle it.
    """
    if not existing_text or not candidate_text:
        return {"conflict_type": "addition", "reason": "empty-input"}
    data = await complete_json(
        system=(
            "You classify whether a new fact contradicts, updates, or "
            "complements an existing fact. Return JSON only."
        ),
        user_message=_CONFLICT_PROMPT.format(
            existing=escape_braces(existing_text, limit=300),
            candidate=escape_braces(candidate_text, limit=300),
        ),
        model=settings.MEM0_CONFLICT_MODEL,
        max_tokens=settings.MEM0_CONFLICT_MAX_TOKENS,
        caller="conflict.classify_conflict",
    )
    if not isinstance(data, dict):
        return {"conflict_type": "addition", "reason": "fail-open"}
    ctype = data.get("conflict_type", "addition")
    if ctype not in VALID_CONFLICT_TYPES:
        return {"conflict_type": "addition", "reason": f"unknown-type:{ctype}"}
    return {"conflict_type": ctype, "reason": str(data.get("reason", ""))[:80]}
