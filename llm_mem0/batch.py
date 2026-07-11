"""Per-channel turn accumulator for batch memory ingestion.

Instead of writing to the memory store every turn, turns are accumulated on
disk and flushed as a single batch when the count reaches
``settings.MEM0_BATCH_SIZE`` (default 10). This reduces LLM extraction calls
from N to 1 per batch window.

State files live under ``.state/mem0_batch/{channel_id}.json`` and survive
process restarts. The next session picks up where the previous one left off.

Concurrency: every mutation is a read-modify-write of the JSON file.
``atomic_write_json`` prevents torn files but NOT lost updates, so all
mutations are serialized under a process-wide lock (callers invoke these
sync functions directly or via ``asyncio.to_thread``; either way the lock
covers them). Cross-process ingestion should stick to one writer process.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from .atomic_write import atomic_write_json
from .settings import settings

log = logging.getLogger(__name__)

# Back-compat module constant (frozen at import); live value is
# settings.MEM0_BATCH_SIZE.
BATCH_SIZE = settings.MEM0_BATCH_SIZE

# Per-turn storage caps: protect the batch file from one pathological turn.
# build_batch_summary applies the (smaller) prompt-facing cap on read.
_TURN_USER_MAX_CHARS = 2000
_TURN_ASSISTANT_MAX_CHARS = 3000

# Serializes load→mutate→save across threads (lost-update guard).
_batch_lock = threading.Lock()


def _batch_dir(state_dir: str | os.PathLike) -> Path:
    d = Path(state_dir) / "mem0_batch"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _batch_file(state_dir: str | os.PathLike, channel_id: int) -> Path:
    return _batch_dir(state_dir) / f"{channel_id}.json"


def _load_raw(state_dir: str | os.PathLike, channel_id: int) -> dict:
    p = _batch_file(state_dir, channel_id)
    if not p.is_file():
        return {"turns": [], "extracted_facts": []}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"turns": [], "extracted_facts": []}
        return data
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("batch load failed for ch=%d: %s", channel_id, exc)
        return {"turns": [], "extracted_facts": []}


def _save_raw(state_dir: str | os.PathLike, channel_id: int, data: dict) -> None:
    p = _batch_file(state_dir, channel_id)
    atomic_write_json(p, data)


def append_turn(
    state_dir: str | os.PathLike,
    channel_id: int,
    user_text: str,
    assistant_text: str,
) -> int:
    """Append a conversation turn and return the updated count."""
    with _batch_lock:
        data = _load_raw(state_dir, channel_id)
        turns = data.get("turns", [])
        turns.append({
            "user": user_text[:_TURN_USER_MAX_CHARS],
            "assistant": assistant_text[:_TURN_ASSISTANT_MAX_CHARS],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        data["turns"] = turns
        _save_raw(state_dir, channel_id, data)
        return len(turns)


def build_batch_summary(turns: list[dict], turn_chars: int | None = None) -> str:
    """Render accumulated turns into the ``user: …\\nassistant: …`` summary fed
    to the extractor. Each turn's user/assistant text is capped at
    ``turn_chars`` (default ``MEM0_BATCH_TURN_CHARS``). Kept here, pure and
    testable, instead of inline in the caller so the truncation budget is one
    documented knob rather than a magic ``[:300]``.
    """
    if turn_chars is None:
        turn_chars = settings.MEM0_BATCH_TURN_CHARS
    return "\n".join(
        f"user: {(t.get('user') or '')[:turn_chars]}\n"
        f"assistant: {(t.get('assistant') or '')[:turn_chars]}"
        for t in (turns or [])
    )


def append_extracted_facts(
    state_dir: str | os.PathLike,
    channel_id: int,
    facts: list[str],
) -> None:
    """Accumulate LLM-extracted facts for batch flush."""
    if not facts:
        return
    with _batch_lock:
        data = _load_raw(state_dir, channel_id)
        existing = data.get("extracted_facts", [])
        existing.extend(facts)
        data["extracted_facts"] = existing
        _save_raw(state_dir, channel_id, data)


def pop_batch(
    state_dir: str | os.PathLike,
    channel_id: int,
) -> tuple[list[dict], list[str]]:
    """Return all accumulated turns + extracted facts and reset the file.

    Returns ``(turns, extracted_facts)``. Atomic w.r.t. the other mutators:
    a turn appended concurrently either lands in this pop or in the next
    batch, never in neither.
    """
    with _batch_lock:
        data = _load_raw(state_dir, channel_id)
        turns = data.get("turns", [])
        extracted_facts = data.get("extracted_facts", [])
        _save_raw(state_dir, channel_id, {"turns": [], "extracted_facts": []})
        return turns, extracted_facts
