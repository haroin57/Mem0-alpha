"""Structured fact layer (Mem0 v5 T8a) — attribute 付き fact の導出インデックス。

gpt-5.5-pro 検算(2026-07-03)の「Chroma=想起、structured layer=truth maintenance」
方針の最小実装。ただし v2 で棄却した「真実の二重化」を避けるため、**fact 本文の
正本はあくまで ChromaDB**。この store は「attribute を持つ fact だけの導出
インデックス」で、壊れたら Chroma から再構築できる(scripts/mem0_backfill_attributes.py)。

役割:
- 決定的 conflict ルートの照会を mem.get_all(全件+Pythonフィルタ) から
  SQL 索引付き lookup に置換（高速・堅牢）
- supersede 連鎖 (supersedes/superseded_by) と valid 期間で属性の履歴を保持
  (「なぜこの値になったか」を event-sourcing の最小形で追える)

接続・スキーマ管理は graph.py の確立パターンを踏襲:
STATE_DIR/mem0_facts.sqlite, WAL, _db_lock で write 直列化, CREATE TABLE IF NOT
EXISTS, 呼び出しは asyncio.to_thread 経由。
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id                 TEXT PRIMARY KEY,   -- ChromaDB row id (正本への参照)
    user_id                 TEXT NOT NULL,
    attribute               TEXT NOT NULL,
    value_text              TEXT,               -- fact 本文 (表示・debug 用コピー)
    value_json              TEXT,               -- typed value (T7, JSON)
    condition               TEXT,               -- time:/context:/custom: or ''
    scope                   TEXT,               -- global/conversation:<id>/meta
    valid_from              INTEGER,
    valid_to                INTEGER,            -- NULL = 現在有効
    status                  TEXT NOT NULL DEFAULT 'active',  -- active/superseded
    source_conversation_id  TEXT,
    extractor_version       TEXT,
    supersedes              TEXT,               -- 置き換えた旧 fact_id
    superseded_by           TEXT,               -- これを置き換えた新 fact_id
    created_at              INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_user_attr
    ON facts(user_id, attribute, status);
CREATE INDEX IF NOT EXISTS idx_facts_user_valid
    ON facts(user_id, valid_to);
"""

_db_lock = threading.Lock()
_db_path: str | None = None


def _resolve_db_path() -> str:
    global _db_path
    if _db_path is None:
        from .settings import STATE_DIR
        os.makedirs(STATE_DIR, exist_ok=True)
        _db_path = os.path.join(STATE_DIR, "mem0_facts.sqlite")
    return _db_path


def _reset_for_tests(path: str | None = None) -> None:
    """テスト用: DB パスを差し替える（本番コードから呼ばない）。"""
    global _db_path
    _db_path = path


@contextmanager
def _conn():
    """Open a connection with WAL, ensure schema, hand it out."""
    path = _resolve_db_path()
    # check_same_thread=False — bot.py + scripts call from various threads
    # via asyncio.to_thread. _db_lock serializes writes; reads are safe.
    c = sqlite3.connect(path, check_same_thread=False)
    c.row_factory = sqlite3.Row
    try:
        c.execute("PRAGMA journal_mode=WAL")
        with _db_lock:
            c.executescript(_SCHEMA)
        yield c
    finally:
        c.close()


def _row_to_dict(r: sqlite3.Row) -> dict:
    return dict(r)


def record_fact(
    fact_id: str,
    user_id: str,
    attribute: str,
    *,
    value_text: str = "",
    value_json: str | None = None,
    condition: str = "",
    scope: str = "",
    source_conversation_id: str | None = None,
    extractor_version: str | None = None,
    supersedes: str | None = None,
) -> bool:
    """attribute 付き fact を登録（INSERT OR REPLACE — 再 ingest に冪等）。

    supersedes を渡すと旧 fact 側の superseded_by / status も同時更新する。
    失敗は False（呼び元は best-effort、正本の Chroma insert は既に済んでいる）。
    """
    if not fact_id or not user_id or not attribute:
        return False
    now = int(time.time())
    try:
        with _conn() as c, _db_lock:
            c.execute(
                """INSERT OR REPLACE INTO facts
                   (fact_id, user_id, attribute, value_text, value_json,
                    condition, scope, valid_from, valid_to, status,
                    source_conversation_id, extractor_version,
                    supersedes, superseded_by, created_at)
                   VALUES (?,?,?,?,?,?,?,?,NULL,'active',?,?,?,NULL,?)""",
                (fact_id, user_id, attribute, value_text[:500], value_json,
                 condition, scope, now,
                 source_conversation_id, extractor_version, supersedes, now),
            )
            if supersedes:
                c.execute(
                    """UPDATE facts SET superseded_by=?, status='superseded',
                       valid_to=? WHERE fact_id=? AND superseded_by IS NULL""",
                    (fact_id, now, supersedes),
                )
            c.commit()
        return True
    except Exception as exc:  # noqa: BLE001 - 導出インデックスは本流を壊さない
        log.warning("fact_store record failed (non-fatal): %s", exc)
        return False


def supersede_fact(old_fact_id: str, new_fact_id: str | None = None) -> bool:
    """旧 fact を supersede 状態にする（Chroma 側 soft-archive と対で呼ぶ）。

    new_fact_id が None のときは「置き換え先不明の失効」（手動 forget 等）。
    """
    if not old_fact_id:
        return False
    now = int(time.time())
    try:
        with _conn() as c, _db_lock:
            c.execute(
                """UPDATE facts SET status='superseded', valid_to=?,
                   superseded_by=COALESCE(?, superseded_by)
                   WHERE fact_id=?""",
                (now, new_fact_id, old_fact_id),
            )
            c.commit()
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("fact_store supersede failed (non-fatal): %s", exc)
        return False


def get_active_facts(user_id: str, attribute: str) -> list[dict]:
    """user × attribute の現在有効な fact 行（決定ルートの照会用）。"""
    if not user_id or not attribute:
        return []
    try:
        with _conn() as c:
            rows = c.execute(
                """SELECT * FROM facts
                   WHERE user_id=? AND attribute=? AND status='active'
                   ORDER BY created_at DESC""",
                (user_id, attribute),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        log.warning("fact_store get_active failed (non-fatal): %s", exc)
        return []


def get_history(user_id: str, attribute: str, limit: int = 50) -> list[dict]:
    """user × attribute の全履歴（supersede 連鎖含む、新しい順）。"""
    if not user_id or not attribute:
        return []
    try:
        with _conn() as c:
            rows = c.execute(
                """SELECT * FROM facts
                   WHERE user_id=? AND attribute=?
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id, attribute, limit),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        log.warning("fact_store get_history failed (non-fatal): %s", exc)
        return []


def count_rows() -> int:
    """weekly compact の突合チェック用（fact_store 側の総数）。"""
    try:
        with _conn() as c:
            row = c.execute("SELECT COUNT(*) AS n FROM facts").fetchone()
        return int(row["n"]) if row else 0
    except Exception:  # noqa: BLE001
        return 0
