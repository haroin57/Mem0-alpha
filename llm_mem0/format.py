"""Format retrieved memories / history hits into an untrusted-context block."""

import re
from datetime import datetime, timezone

from .llm import strip_json_fence  # noqa: F401 — re-export (moved to llm.py)
from .sanitize import POWER_PHRASE_RE

# Markers an attacker could have stored in a fact (via the ingestion path)
# that would close the <retrieved_memories> fence or impersonate any trusted
# prompt-scaffold block when echoed back into the prompt. Strip them
# defensively at format time as second-order injection defense.
#
# This pattern set MIRRORS `_SENTINEL_RE` in bot.py / services/prompt_builder.py
# so a fact echoed via format_memories_for_prompt cannot inject a scaffold
# marker that the outer sanitize layer used to catch. The outer
# `_sanitize_untrusted_text(mem0_block)` call was removed for hot-path
# performance, so this blocklist is now the single source of scrubbing.
_FACT_TEXT_BLOCKLIST = re.compile(
    r"\[(?:Message from|End of|Mem0|Attached images|Current message|Thread:|"
    r"Channel recent|Referenced message|Replied-to message|End of thread|"
    r"End of channel|End of Mem0|History|End of history|Trusted context|"
    r"untrusted_user_input)"
    r"[^\]\n]*\]"
    r"|</?retrieved_memories[^>]*>"
    r"|</?history_search[^>]*>"
    r"|</?untrusted_user_input[^>]*>",
    re.IGNORECASE,
)


def _scrub_fact_text(text: str) -> str:
    """Drop markers that could close the memories fence or impersonate trust headers.

    Also applies POWER_PHRASE_RE to redact power-grab phrases (M15).
    """
    if not text:
        return text
    text = _FACT_TEXT_BLOCKLIST.sub("[redacted-marker]", text)
    text = POWER_PHRASE_RE.sub("[redacted]", text)
    return text


_TIER_LABELS = {0: "Live", 1: "Hot", 2: "Warm", 3: "Cold"}


def _format_epoch(ts) -> str | None:
    """unix 秒を YYYY-MM-DD に。壊れた値は None（メタ無し扱い）。"""
    try:
        ts = int(ts)
        if ts <= 0:
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _fact_meta_suffix(m: dict) -> str:
    """事実1行に付ける [imp=N Tier] [YYYY-MM-DD] サフィックス。

    importance / tier / 日付をモデルに見せることで、重要度5のコア事実と
    重要度2の断片、新しい状態と古い状態をモデルが区別できるようにする
    （従来は全メタデータを捨てて全事実が等価に見えていた）。
    メタが無い古いレコードは空文字を返し、従来表示のままになる。
    """
    md = m.get("metadata") or {}
    if not isinstance(md, dict):
        return ""
    bits: list[str] = []
    try:
        imp = int(md.get("importance"))
    except (TypeError, ValueError):
        imp = None
    tier_label = _TIER_LABELS.get(md.get("tier"))
    inner = " ".join(
        s for s in (f"imp={imp}" if imp is not None else "", tier_label or "")
        if s
    )
    if inner:
        bits.append(f"[{inner}]")
    date_str = _format_epoch(
        md.get("last_reinforced_at") or md.get("created_at_unix")
        or md.get("timestamp")
    )
    if date_str:
        bits.append(f"[{date_str}]")
    # Phase R (Mem0 v5 T2): 保存時に構造化した condition/scope を注入時に
    # 落とさない（flattening 防止）。「夜は紅茶」が条件を失って「紅茶が好き」
    # として無条件注入されるのを防ぐ。metadata が無ければ何も足さない。
    cond = md.get("condition")
    if isinstance(cond, str) and cond.strip():
        bits.append(f"[条件: {_scrub_fact_text(cond.strip()[:48])}]")
    scope = md.get("scope")
    if isinstance(scope, str):
        if scope.startswith("conversation"):
            bits.append("[この会話限定]")
        elif scope == "meta":
            bits.append("[メタ情報]")
    return (" " + " ".join(bits)) if bits else ""


def format_memories_for_prompt(memories: list[dict]) -> str:
    """Format retrieved memories as a prompt-injection block.

    Phase ⑦: a ``_fok_no_memory`` sentinel (see
    ``search.assess_recall_confidence`` / ``MEM0_FOK_ENABLED``) renders as an
    explicit "nothing relevant found" line rather than being silently
    dropped — a calibrated null result reads very differently to the model
    than the fact block simply not being there, and the latter invites
    confabulation ("memory wasn't consulted" vs. "memory was consulted and
    came up empty"). Core memories (unconditional, query-independent) still
    render normally alongside it when present.
    """
    if not memories:
        return ""

    fok_entry = next((m for m in memories if m.get("_fok_no_memory")), None)
    real = [m for m in memories if not m.get("_fok_no_memory")]
    if not real and not fok_entry:
        return ""

    lines = [
        "[Long-term memory — facts about this user]",
        "(legend: imp=importance 1-5 / Live>Hot>Warm>Cold=freshness tier / "
        "[date]=last confirmed / [条件: ...]=only holds under that condition)",
    ]
    # core memory（常時注入枠）を先頭に固定し、クエリ依存の recall と区別する。
    ordered = sorted(
        real, key=lambda m: 0 if m.get("_core") else 1,
    )
    for m in ordered:
        text = m.get("memory", m.get("text", ""))
        if text:
            lines.append(f"- {_scrub_fact_text(text)}{_fact_meta_suffix(m)}")
    if fok_entry:
        lines.append(
            "(このクエリに一致する具体的な記憶は見つからなかった。"
            "推測で補完しないこと)"
        )
    lines.append("[End of Long-term memory]")
    return "\n".join(lines)


def format_history_for_prompt(results: list[dict]) -> str:
    """Format BM25 history-search hits as a prompt-injection block.

    Mirrors format_memories_for_prompt: same defensive _scrub_fact_text on the
    verbatim line so a stored message can't close the <history_search> fence.
    Each hit is labelled with its role (発言者: user=この人 / assistant=私).
    """
    if not results:
        return ""

    lines = ["[History — past log lines that matched the query keywords]"]
    for r in results:
        text = r.get("memory") or r.get("text") or ""
        if not text:
            continue
        role = r.get("role", "")
        author = (r.get("author") or "").strip()
        if role == "user":
            # history_search はチャンネル過去ログから keyword 一致で行を拾う。
            # 複数の admin が同居するチャンネルでは、現発言者とは別人の過去発言も
            # 混ざる。"この人" は Mem0 ブロック ([Mem0 — この人についての…]) と同語で
            # 現発言者の情報だと誤読させるため避ける。author が分かればそれを使い、
            # 不明なら話者非断定の中立ラベルにフォールバックする (author 列は
            # JSONL author 保存の修正後に実名化される)。
            who = author or "過去の発言者"
        elif role == "assistant":
            who = "私"
        elif role == "channel":
            # Non-mention channel message — the speaker can be anyone, so show
            # the author display name when known.
            who = author or "誰か"
        else:
            who = author or role
        lines.append(f"- ({who}) {_scrub_fact_text(text)}")
    if len(lines) == 1:
        return ""
    lines.append("[End of history]")
    return "\n".join(lines)
