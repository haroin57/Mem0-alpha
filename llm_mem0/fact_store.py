"""Structured fact layer (Mem0 v5 T8a) — attribute 付き fact の導出インデックス。

「Chroma=想起、structured layer=truth maintenance」方針の最小実装。ただし
「真実の二重化」を避けるため、**fact 本文の正本はあくまで ChromaDB**。この
store は「attribute を持つ fact だけの導出インデックス」で、壊れたら Chroma
から再構築できる。

役割:
- 決定的 conflict ルートの照会を mem.get_all(全件+Pythonフィルタ) から
  SQL 索引付き lookup に置換（高速・堅牢）
- supersede 連鎖 (supersedes/superseded_by) と valid 期間で属性の履歴を保持
  (「なぜこの値になったか」を event-sourcing の最小形で追える)

接続・スキーマ管理は :mod:`llm_mem0.sqlite_store` に集約。呼び出しは
asyncio.to_thread 経由。
"""

from __future__ import annotations

import logging
import time

from .sqlite_store import SqliteStore, row_to_dict

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

# 表示・debug 用コピーの上限。正本は Chroma 側なので切り詰めは非破壊。
_VALUE_TEXT_MAX_CHARS = 500

_store = SqliteStore("mem0_facts.sqlite", _SCHEMA)


def _reset_for_tests(path: str | None = None) -> None:
    """テスト用: DB パスを差し替える（本番コードから呼ばない）。"""
    _store.reset_path_for_test(path)


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

    def _record(c) -> bool:
        c.execute(
            """INSERT OR REPLACE INTO facts
               (fact_id, user_id, attribute, value_text, value_json,
                condition, scope, valid_from, valid_to, status,
                source_conversation_id, extractor_version,
                supersedes, superseded_by, created_at)
               VALUES (?,?,?,?,?,?,?,?,NULL,'active',?,?,?,NULL,?)""",
            (fact_id, user_id, attribute, value_text[:_VALUE_TEXT_MAX_CHARS],
             value_json, condition, scope, now,
             source_conversation_id, extractor_version, supersedes, now),
        )
        if supersedes:
            c.execute(
                """UPDATE facts SET superseded_by=?, status='superseded',
                   valid_to=? WHERE fact_id=? AND superseded_by IS NULL""",
                (fact_id, now, supersedes),
            )
        return True

    return bool(_store.write(_record, what="record_fact", default=False))


def supersede_fact(old_fact_id: str, new_fact_id: str | None = None) -> bool:
    """旧 fact を supersede 状態にする（Chroma 側 soft-archive と対で呼ぶ）。

    new_fact_id が None のときは「置き換え先不明の失効」（手動 forget 等）。
    """
    if not old_fact_id:
        return False
    now = int(time.time())

    def _supersede(c) -> bool:
        c.execute(
            """UPDATE facts SET status='superseded', valid_to=?,
               superseded_by=COALESCE(?, superseded_by)
               WHERE fact_id=?""",
            (now, new_fact_id, old_fact_id),
        )
        return True

    return bool(_store.write(_supersede, what="supersede_fact", default=False))


def get_active_facts(user_id: str, attribute: str) -> list[dict]:
    """user × attribute の現在有効な fact 行（決定ルートの照会用）。"""
    if not user_id or not attribute:
        return []

    def _fetch(c) -> list[dict]:
        rows = c.execute(
            """SELECT * FROM facts
               WHERE user_id=? AND attribute=? AND status='active'
               ORDER BY created_at DESC""",
            (user_id, attribute),
        ).fetchall()
        return [row_to_dict(r) for r in rows]

    return _store.read(_fetch, what="get_active_facts", default=[]) or []


def get_history(user_id: str, attribute: str, limit: int = 50) -> list[dict]:
    """user × attribute の全履歴（supersede 連鎖含む、新しい順）。"""
    if not user_id or not attribute:
        return []

    def _fetch(c) -> list[dict]:
        rows = c.execute(
            """SELECT * FROM facts
               WHERE user_id=? AND attribute=?
               ORDER BY created_at DESC LIMIT ?""",
            (user_id, attribute, limit),
        ).fetchall()
        return [row_to_dict(r) for r in rows]

    return _store.read(_fetch, what="get_history", default=[]) or []


def count_rows() -> int:
    """weekly compact の突合チェック用（fact_store 側の総数）。

    -1 = 照会自体の失敗。「0 件」と区別できるようセンチネルを返す
    （突合レポートが空 DB と障害を混同しないため）。
    """
    result = _store.read(
        lambda c: int(c.execute("SELECT COUNT(*) AS n FROM facts").fetchone()["n"]),
        what="count_rows", default=-1,
    )
    return result if result is not None else -1
