"""Two-stage dedup: cosine threshold + LLM judgment for gray zone."""

from __future__ import annotations

import logging

from .llm import complete_json, escape_braces
from .settings import settings

log = logging.getLogger(__name__)

# mem0's Chroma adapter returns `score = distance` (NOT similarity).
# ChromaDB default metric is L2; with OpenAI normalized embeddings this lands
# roughly in [0, 2]: 0 = identical text, ~1.0 = unrelated, ~1.5+ = opposite
# structure. So small distance ⇒ duplicate, large distance ⇒ novel.
#
# Observed examples (illustrative):
#   "Alice の音楽が好き" vs "Bob の音楽が好き"             → 0.817 (related, diff entity)
#   "Alice の音楽が好き" vs "Carol が好き"                → 0.991 (unrelated)
#   "Alice のライブに行った" vs "Bob の楽曲"              → 1.296 (unrelated)
DEDUP_HARD_DROP_DISTANCE = 0.15   # distance <= this → confirmed duplicate (drop)
DEDUP_GRAY_HIGH_DISTANCE = 0.35   # distance in (HARD, GRAY] → LLM judges
# distance > GRAY_HIGH → confirmed novel, keep

# Legacy aliases kept so any external import doesn't break. DO NOT use in
# new code — semantics are now distance-based.
DEDUP_HARD_DROP = DEDUP_HARD_DROP_DISTANCE
DEDUP_GRAY_LOW = DEDUP_GRAY_HIGH_DISTANCE

# Three-way relation verdict (replaces the old binary same/different judge).
# The old design collapsed "pure paraphrase" and "superset/complementary" into
# a single "same" verdict that dropped the NEW candidate — silently discarding
# the richer fact when the candidate added detail. Splitting them lets us drop
# only on true paraphrase and *synthesize* (止揚) on mergeable overlap.
_DEDUP_RELATION_PROMPT = """以下の2つの文の関係を判定してください。

既存の記憶: {existing}
新しい候補: {candidate}

判定 (3択、迷ったら "different"):
- "paraphrase": 同じ事実・同じ側面の単なる言い換え。情報の増減なし。
  - 「Rust が好き」vs「Rust を好んでいる」
  - 「dev ブランチに直接 push 禁止」vs「dev ブランチへの直接 push は禁止」
- "mergeable": 同じ事実・同じ側面だが、片方が日付/数値/固有名詞/限定詞/理由などの
  具体情報を追加している (上位互換または相互補完)。統合すべき。
  - 「大学からプログラミングを始めた」vs「大学一年から独学でプログラミングを始めた」
  - 「Denon AH-D5200 の購入を決定 (時期未定)」vs「Denon AH-D5200 を6/20購入決定、リスニング専用」
- "different": 別の側面・別の事実。両方を独立に保持すべき。
  - 「Rust を業務で使ってる」(行動) vs「Rust が好き」(嗜好)
  - 「Blade Runner を知ってるか質問した」(会話) vs「Blade Runner が好き」(嗜好)
  - 「取り返しのつかなさを描く作品が好き」(抽象) vs「Blade Runner が好き」(固有名詞)

誤統合より独立保持が安全。少しでも別の側面に見えたら "different"。

以下のJSONのみ返してください(説明不要):
{{"relation": "paraphrase" | "mergeable" | "different", "reason": "<30 文字以内>"}}"""

_DEDUP_MERGE_PROMPT = """次の2つの記憶は同じ事実を別の粒度・側面で述べています。
両方が持つ具体情報 (日付・数値・固有名詞・限定詞・理由) を一つも落とさずに
1文へ統合してください。

- 状態が変化している場合 (例: 未定→確定、旧値→新値) は新しい方の状態を採用し、
  古い状態は書かない。
- 冗長な重複表現は1回にまとめる。
- 主語・三人称/一人称・語尾は既存記憶の文体に合わせる。

既存の記憶: {existing}
新しい候補: {candidate}

以下のJSONのみ返してください(説明不要):
{{"merged": "<統合した1文>"}}"""

_VALID_RELATIONS = ("paraphrase", "mergeable", "different")


async def judge_fact_relation(existing_text: str, candidate_text: str) -> str:
    """Classify the relation between two facts.

    Returns one of ``paraphrase`` / ``mergeable`` / ``different``.
    On any error, returns ``different`` (fail-open: keep both, never drop
    or merge on a judge failure).
    """
    data = await complete_json(
        system=(
            "You classify the relation between two short fact statements "
            "as paraphrase / mergeable / different. Return JSON only."
        ),
        user_message=_DEDUP_RELATION_PROMPT.format(
            existing=escape_braces(existing_text, limit=300),
            candidate=escape_braces(candidate_text, limit=300),
        ),
        model=settings.MEM0_DEDUP_MODEL,
        max_tokens=settings.MEM0_DEDUP_MAX_TOKENS,
        caller="dedup.judge_fact_relation",
    )
    relation = (data or {}).get("relation") if isinstance(data, dict) else None
    return relation if relation in _VALID_RELATIONS else "different"


async def llm_judge_same_fact(existing_text: str, candidate_text: str) -> bool:
    """Back-compat shim: True when the two facts are not independent.

    Retained for callers/tests that want the old binary signal. Internally
    delegates to ``judge_fact_relation``; ``paraphrase``/``mergeable`` map to
    True (not independent), ``different`` maps to False. On error the relation
    judge fails open to ``different`` → returns False (keep the candidate).
    """
    return (await judge_fact_relation(existing_text, candidate_text)) != "different"


async def synthesize_facts(existing_text: str, candidate_text: str) -> str | None:
    """Aufhebung (止揚): merge two overlapping facts into one richer fact.

    Returns the synthesized fact text, or ``None`` on any error (fail-open:
    the caller keeps both facts rather than losing information).
    """
    data = await complete_json(
        system=(
            "You merge two overlapping fact statements into a single fact "
            "that preserves every concrete detail. Return JSON only."
        ),
        user_message=_DEDUP_MERGE_PROMPT.format(
            existing=escape_braces(existing_text, limit=400),
            candidate=escape_braces(candidate_text, limit=400),
        ),
        model=settings.MEM0_DEDUP_MERGE_MODEL,
        max_tokens=settings.MEM0_DEDUP_MAX_TOKENS,
        caller="dedup.synthesize_facts",
    )
    merged = (data or {}).get("merged") if isinstance(data, dict) else None
    if isinstance(merged, str) and merged.strip():
        return merged.strip()
    return None


_CLUSTER_CONSOLIDATE_PROMPT = """次の複数の記憶は同じ話題について別々のタイミングで保存されたものです。
重複・言い換え・進行状況の更新が混在しています。これらを統合してください。

規則:
- 同一事実の言い換え・重複は1つにまとめる
- 時系列で状態が進行している場合（例: 進捗、未定→確定）は最新状態を採用し、古い途中経過は捨てる
- 具体情報（日付・数値・固有名詞・ルート名など）は最新性を保ちつつ漏らさず保持
- **別々に保つべき独立した事実（別の側面）が混ざっていれば分けて出力してよい**（無理に1つにしない）
- 出力件数は元より必ず少なくする（最低でも1件減らせない場合は統合しない）

記憶:
{facts}

以下のJSONのみ返してください(説明不要):
{{"consolidated": ["統合後fact1", ...], "note": "<30字以内>"}}"""


async def synthesize_cluster(texts: list[str]) -> list[str] | None:
    """N-way consolidation (止揚) of a related-memory cluster.

    Returns the consolidated fact list (always SHORTER than the input on a
    successful consolidation), or ``None`` on any error / no-op (fail-open:
    the caller leaves the cluster untouched). Empirically validated: a 7-fact
    temporal cluster about one topic collapses to the single latest-state fact.

    Distinct aspects accidentally clustered together are split back out by the
    LLM (the prompt allows >1 output), so over-consolidation is bounded.
    """
    facts = [t for t in (texts or []) if isinstance(t, str) and t.strip()]
    if len(facts) < 2:
        return None
    data = await complete_json(
        system=(
            "You consolidate a cluster of redundant memory facts into "
            "fewer facts, preserving every concrete detail and the latest "
            "state. Return JSON only."
        ),
        user_message=_CLUSTER_CONSOLIDATE_PROMPT.format(
            facts="\n".join(f"- {escape_braces(t, limit=300)}" for t in facts),
        ),
        model=settings.MEM0_DEDUP_MERGE_MODEL,
        max_tokens=settings.MEM0_DEDUP_MAX_TOKENS,
        caller="dedup.synthesize_cluster",
    )
    consolidated = (data or {}).get("consolidated") if isinstance(data, dict) else None
    if not isinstance(consolidated, list):
        return None
    consolidated = [c.strip() for c in consolidated if isinstance(c, str) and c.strip()]
    # Must actually reduce the count; otherwise it's a no-op (skip).
    if not consolidated or len(consolidated) >= len(facts):
        return None
    return consolidated


async def partition_mergeable(texts: list[str]) -> list[list[int]]:
    """Split a candidate cluster into sub-clusters that are safe to merge.

    Over-merge guard for offline consolidation (ADR-0017 / P3): vector
    clustering can group facts that share an entity but describe DISTINCT
    aspects ("Rust好き" vs "Rustを業務で使ってる"). ``synthesize_cluster``
    on its own over-merges those, so we gate it here using the validated
    ``judge_fact_relation``: run it pairwise, treat ``paraphrase``/``mergeable``
    as an edge, and return the Union-Find connected components. ``different``
    pairs leave the members in separate components, so they are never fused.

    Byte-identical pairs skip the LLM (trivially ``paraphrase``). Returns a
    list of index-lists (each over the input ``texts``); singletons are
    included so callers can decide to leave them untouched.
    """
    n = len(texts)
    if n <= 1:
        return [[i] for i in range(n)]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    norm = [(t or "").strip() for t in texts]
    for i in range(n):
        for j in range(i + 1, n):
            if norm[i] and norm[i] == norm[j]:
                union(i, j)
                continue
            rel = await judge_fact_relation(norm[i], norm[j])
            if rel in ("paraphrase", "mergeable"):
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    # Stable order: by smallest member index.
    return [groups[k] for k in sorted(groups, key=lambda r: min(groups[r]))]


def _log_dedup_decision(
    *, decision: str, cand: str, existing: str,
    distance: float, dry_run: bool,
    llm_same: bool | None = None,
) -> None:
    """One-line structured log so it can be grep'd later for calibration.

    Phase A-3 (mem0-recall-hybrid-hyde-kg): used by `journalctl | grep
    DEDUP_DECISION` to spot under/over-dropping over a 1-week window
    before deciding whether to retune DEDUP_HARD_DROP_DISTANCE /
    DEDUP_GRAY_HIGH_DISTANCE.
    """
    log.info(
        "[DEDUP_DECISION decision=%s dry_run=%s distance=%.3f llm_same=%s "
        "cand=%r existing=%r]",
        decision, dry_run, distance, llm_same,
        cand[:80], existing[:80],
    )


async def check_near_dup(
    candidate_text: str,
    candidate_category: str | None,
    near_results: list[dict],
    *,
    dry_run: bool = False,
    detect_conflict: bool = False,
    allow_merge: bool = True,
) -> tuple[bool, str, str | None, str | None, str | None]:
    """Two-stage dedup with a 止揚 (aufhebung) merge path for supersets.

    Returns ``(is_dup, reason, existing_id, conflict_type, merged_text)``.
    - is_dup=True  → candidate is a true duplicate, should be dropped;
      ``existing_id`` is the matched memory (caller reinforces its
      mention_count). conflict_type/merged_text are None.
    - is_dup=False + conflict_type="merge" + merged_text set → the candidate
      and the existing fact overlap with complementary detail; the caller
      inserts ``merged_text`` (not the raw candidate) and soft-archives
      ``existing_id`` (止揚: preserve both, elevate into one richer fact).
    - is_dup=False + conflict_type in ("update","correction") → state change /
      fix; caller keeps the candidate as-is and archives ``existing_id``.
    - is_dup=False + conflict_type=None → keep the candidate independently.

    Relation in the gray zone:
      paraphrase → drop (reinforce existing), mergeable → synthesize,
      different → conflict detection then keep. Hard-drop (≤0.15) stays a
      pure-duplicate drop — those distances are exact paraphrases.

    ``dry_run=True`` computes the full decision (including the LLM calls so we
    measure agreement) but never drops/merges; reasons are prefixed
    ``dry_run_would_``. ``allow_merge=False`` restores the legacy behavior
    where a mergeable superset is dropped like a paraphrase.
    """
    for r in near_results:
        score = r.get("score")
        if score is None:
            # A scoreless neighbor (e.g. graph-expanded rows before their
            # placeholder is set) carries no distance signal. Treating it as
            # distance 0 would hard-drop the candidate as an exact duplicate;
            # skip the neighbor instead (fail-open: keep the candidate).
            continue
        existing_text = r.get("memory", "")
        existing_id = r.get("id")

        # Category filter: skip dedup against different-category memories
        existing_category = (r.get("metadata") or {}).get("category")
        if candidate_category and existing_category and candidate_category != existing_category:
            continue

        # score is *distance* (small = similar). Sort-by-distance ascending
        # means the first iteration already has the smallest distance, but
        # we keep iterating just in case mem0 ever changes ordering.
        if score <= DEDUP_HARD_DROP_DISTANCE:
            _log_dedup_decision(
                decision="hard_drop", cand=candidate_text,
                existing=existing_text, distance=score, dry_run=dry_run,
            )
            log.info(
                "dedup HARD_DROP: cand='%s' | existing='%s' | distance=%.3f",
                candidate_text[:100], existing_text[:100], score,
            )
            if dry_run:
                return (False, f"dry_run_would_hard_drop(distance={score:.3f})", existing_id, None, None)
            return (True, f"hard_drop(distance={score:.3f})", existing_id, None, None)

        if score <= DEDUP_GRAY_HIGH_DISTANCE:
            relation = await judge_fact_relation(existing_text, candidate_text)
            decision = {
                "paraphrase": "gray_paraphrase",
                "mergeable": "gray_merge",
                "different": "gray_kept",
            }.get(relation, "gray_kept")
            _log_dedup_decision(
                decision=decision, cand=candidate_text, existing=existing_text,
                distance=score, dry_run=dry_run,
                llm_same=(relation != "different"),
            )
            log.info(
                "dedup GRAY: cand='%s' | existing='%s' | distance=%.3f | relation=%s",
                candidate_text[:100], existing_text[:100], score, relation,
            )

            if relation == "paraphrase":
                if dry_run:
                    return (False, f"dry_run_would_llm_drop(distance={score:.3f})", existing_id, None, None)
                return (True, f"paraphrase(distance={score:.3f})", existing_id, None, None)

            if relation == "mergeable" and allow_merge:
                if dry_run:
                    return (False, f"dry_run_would_merge(distance={score:.3f})", existing_id, None, None)
                merged = await synthesize_facts(existing_text, candidate_text)
                if merged:
                    log.info(
                        "[DEDUP_MERGE existing_id=%s distance=%.3f merged=%r]",
                        existing_id, score, merged[:100],
                    )
                    return (False, f"merged(distance={score:.3f})",
                            existing_id, "merge", merged)
                # synthesis failed → keep both rather than lose the candidate
                return (False, f"merge_failed_keep_both(distance={score:.3f})", None, None, None)

            if relation == "mergeable" and not allow_merge:
                # Legacy behavior: drop the superset like a paraphrase.
                if dry_run:
                    return (False, f"dry_run_would_llm_drop(distance={score:.3f})", existing_id, None, None)
                return (True, f"merge_disabled_drop(distance={score:.3f})", existing_id, None, None)

            # relation == "different": gray-zone but a distinct aspect. Check
            # whether it is actually a state change / fix of the existing fact
            # (update / correction → archive old, keep new). First gray hit is
            # the closest cosine match and the most likely contradiction target.
            if detect_conflict:
                from .conflict import classify_conflict
                cls = await classify_conflict(existing_text, candidate_text)
                ctype = cls.get("conflict_type", "addition")
                if ctype in ("update", "correction"):
                    log.info(
                        "[CONFLICT_GRAY conflict_type=%s reason=%r existing_id=%s "
                        "distance=%.3f]",
                        ctype, cls.get("reason", ""), existing_id, score,
                    )
                    return (False, f"conflict_{ctype}(distance={score:.3f})",
                            existing_id, ctype, None)

    return (False, "novel", None, None, None)
