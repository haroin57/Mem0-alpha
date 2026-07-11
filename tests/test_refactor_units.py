"""Unit tests for the refactored internals (no network, no real LLM).

Covers: dynamic settings, the shared LLM-JSON helper, the shared SQLite
store (WAL-once + lock semantics), store roundtrips, the decomposed ingest
helpers, the batch lock, and the MCP tool error paths.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

import llm_mem0.settings as settings_mod
from llm_mem0.settings import settings


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
class TestSettings:
    def test_defaults(self):
        assert settings.MEM0_HYBRID_ENABLED is True
        assert settings.MEM0_GATE_MIN_IMPORTANCE == 2
        assert settings.MEM0_EXTRACT_MAX_TOKENS == 2000
        assert settings.CHROMA_MODE == "server"

    def test_dynamic_reads(self, monkeypatch):
        monkeypatch.setenv("MEM0_HYBRID_ENABLED", "false")
        assert settings.MEM0_HYBRID_ENABLED is False
        monkeypatch.setenv("MEM0_BM25_LIMIT", "7")
        assert settings.MEM0_BM25_LIMIT == 7

    def test_bad_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MEM0_BM25_LIMIT", "not-a-number")
        assert settings.MEM0_BM25_LIMIT == 20

    def test_unknown_setting_raises(self):
        with pytest.raises(AttributeError):
            settings.NO_SUCH_SETTING

    def test_pep562_legacy_import(self):
        assert settings_mod.MEM0_GATE_ENABLED is True
        assert settings_mod.MEM0_HYDE_ENABLED is True
        with pytest.raises(AttributeError):
            settings_mod.NO_SUCH_SETTING

    def test_state_dir_created(self, tmp_path, monkeypatch):
        target = tmp_path / "state"
        monkeypatch.setenv("LLM_MEM0_STATE_DIR", str(target))
        assert settings.STATE_DIR == str(target)
        assert target.is_dir()


# ---------------------------------------------------------------------------
# llm helpers
# ---------------------------------------------------------------------------
class TestLlmHelpers:
    def test_escape_braces(self):
        from llm_mem0.llm import escape_braces
        assert escape_braces("a{b}c") == "a{{b}}c"
        assert escape_braces("a{b}c", limit=3) == "a{{b"

    def test_strip_json_fence(self):
        from llm_mem0.llm import strip_json_fence
        assert strip_json_fence('{"a": 1}') == '{"a": 1}'
        assert strip_json_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
        assert strip_json_fence('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_format_reexports_strip_json_fence(self):
        from llm_mem0.format import strip_json_fence as f1
        from llm_mem0.llm import strip_json_fence as f2
        assert f1 is f2

    @pytest.mark.asyncio
    async def test_complete_json_parses(self, monkeypatch):
        from llm_mem0 import llm

        class FakeBackend:
            async def complete(self, *, system, user_message, model, max_tokens):
                return '```json\n{"relation": "different"}\n```'

        monkeypatch.setattr(
            "llm_mem0.auth.get_auth_backend", lambda: FakeBackend())
        out = await llm.complete_json(
            system="s", user_message="u", model="m", max_tokens=10)
        assert out == {"relation": "different"}

    @pytest.mark.asyncio
    async def test_complete_json_fails_open(self, monkeypatch):
        from llm_mem0 import llm

        class EmptyBackend:
            async def complete(self, **kw):
                return ""

        class BrokenBackend:
            async def complete(self, **kw):
                raise RuntimeError("boom")

        class GarbageBackend:
            async def complete(self, **kw):
                return "not json at all"

        for backend in (EmptyBackend(), BrokenBackend(), GarbageBackend()):
            monkeypatch.setattr(
                "llm_mem0.auth.get_auth_backend", lambda b=backend: b)
            out = await llm.complete_json(
                system="s", user_message="u", model="m", max_tokens=10)
            assert out is None


# ---------------------------------------------------------------------------
# dedup / conflict fail-open (through complete_json)
# ---------------------------------------------------------------------------
class TestJudgesFailOpen:
    @pytest.mark.asyncio
    async def test_judge_fact_relation_fail_open(self, monkeypatch):
        class BrokenBackend:
            async def complete(self, **kw):
                raise RuntimeError("no llm")

        monkeypatch.setattr(
            "llm_mem0.auth.get_auth_backend", lambda: BrokenBackend())
        from llm_mem0.dedup import judge_fact_relation, llm_judge_same_fact
        assert await judge_fact_relation("a", "b") == "different"
        assert await llm_judge_same_fact("a", "b") is False

    @pytest.mark.asyncio
    async def test_classify_conflict_fail_open(self, monkeypatch):
        class BrokenBackend:
            async def complete(self, **kw):
                raise RuntimeError("no llm")

        monkeypatch.setattr(
            "llm_mem0.auth.get_auth_backend", lambda: BrokenBackend())
        from llm_mem0.conflict import classify_conflict
        out = await classify_conflict("既存", "新規")
        assert out["conflict_type"] == "addition"

    @pytest.mark.asyncio
    async def test_judge_fact_relation_valid_verdict(self, monkeypatch):
        class Backend:
            async def complete(self, **kw):
                return '{"relation": "mergeable", "reason": "detail added"}'

        monkeypatch.setattr(
            "llm_mem0.auth.get_auth_backend", lambda: Backend())
        from llm_mem0.dedup import judge_fact_relation
        assert await judge_fact_relation("a", "b") == "mergeable"

    @pytest.mark.asyncio
    async def test_check_near_dup_skips_scoreless_neighbor(self):
        from llm_mem0.dedup import check_near_dup
        # A neighbor with score=None must not be treated as distance 0
        # (which would hard-drop the candidate as an exact duplicate).
        is_dup, reason, existing_id, ctype, merged = await check_near_dup(
            "新しい事実", None, [{"score": None, "memory": "既存", "id": "e1"}],
        )
        assert is_dup is False and reason == "novel"


# ---------------------------------------------------------------------------
# sqlite_store
# ---------------------------------------------------------------------------
class TestSqliteStore:
    def test_wal_applied_once_and_roundtrip(self, tmp_path):
        from llm_mem0.sqlite_store import SqliteStore
        store = SqliteStore(
            "t.sqlite", "CREATE TABLE IF NOT EXISTS t (k TEXT PRIMARY KEY, v TEXT);")
        store.reset_path_for_test(str(tmp_path / "t.sqlite"))
        assert store.write(
            lambda c: c.execute("INSERT INTO t VALUES ('a', '1')") and True)
        with store.connect() as c:
            assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        got = store.read(
            lambda c: c.execute("SELECT v FROM t WHERE k='a'").fetchone()["v"])
        assert got == "1"

    def test_write_returns_default_on_error(self, tmp_path):
        from llm_mem0.sqlite_store import SqliteStore
        store = SqliteStore("t.sqlite", "CREATE TABLE IF NOT EXISTS t (k TEXT);")
        store.reset_path_for_test(str(tmp_path / "t.sqlite"))

        def bad(c):
            raise ValueError("nope")

        assert store.write(bad, default=-1) == -1

    def test_wipe_recreates_schema(self, tmp_path):
        from llm_mem0.sqlite_store import SqliteStore
        store = SqliteStore("t.sqlite", "CREATE TABLE IF NOT EXISTS t (k TEXT);")
        store.reset_path_for_test(str(tmp_path / "t.sqlite"))
        store.write(lambda c: c.execute("INSERT INTO t VALUES ('x')"))
        store.wipe()
        n = store.read(lambda c: c.execute("SELECT count(*) FROM t").fetchone()[0])
        assert n == 0

    def test_build_fts_match_query_strips_syntax(self):
        from llm_mem0.sqlite_store import build_fts_match_query
        q = build_fts_match_query('evil (query)^ AND: stuff')
        # FTS5 syntax characters must never survive into the MATCH string;
        # the surviving tokens are OR-joined quoted phrases.
        assert not any(ch in q for ch in "()^:*")
        assert build_fts_match_query("") == ""

    def test_concurrent_writes_no_row_drop(self, tmp_path):
        """The WAL-once + retry pattern must not silently drop rows under
        concurrent writers (the old per-connection-WAL bug lost ~30%)."""
        from llm_mem0.sqlite_store import SqliteStore
        store = SqliteStore(
            "c.sqlite", "CREATE TABLE IF NOT EXISTS t (k INTEGER PRIMARY KEY);")
        store.reset_path_for_test(str(tmp_path / "c.sqlite"))
        n_threads, per_thread = 8, 25
        results = []

        def worker(base):
            for i in range(per_thread):
                ok = store.write(
                    lambda c, v=base * 1000 + i:
                        c.execute("INSERT INTO t VALUES (?)", (v,)) and True,
                    default=False,
                )
                results.append(bool(ok))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(results)
        total = store.read(
            lambda c: c.execute("SELECT count(*) FROM t").fetchone()[0])
        assert total == n_threads * per_thread


# ---------------------------------------------------------------------------
# graph: add_relation id semantics (the INSERT OR IGNORE fix)
# ---------------------------------------------------------------------------
class TestGraph:
    @pytest.fixture(autouse=True)
    def _tmp_db(self, tmp_path):
        from llm_mem0 import graph
        graph._reset_db_path_for_test(str(tmp_path / "graph.sqlite"))
        graph.invalidate_top_entities_cache()
        yield
        graph._reset_db_path_for_test(None)
        graph.invalidate_top_entities_cache()

    def test_add_relation_ignore_returns_existing_id(self):
        from llm_mem0 import graph
        a = graph.upsert_entity("Alice", "person", "u1")
        b = graph.upsert_entity("Rust", "tech", "u1")
        rid1 = graph.add_relation(a, "loves", b, "u1")
        rid2 = graph.add_relation(a, "loves", b, "u1")
        assert rid1 > 0
        assert rid2 == rid1

    def test_alias_lookup_and_merge(self):
        from llm_mem0 import graph
        eid = graph.upsert_entity(
            "Blade Runner", "media", "u1", aliases=["ブレードランナー"])
        again = graph.upsert_entity("ブレードランナー", "media", "u1")
        assert again == eid
        found = graph.find_entity("ブレードランナー", "u1", fuzzy=False)
        assert found is not None and found.canonical_name == "Blade Runner"


# ---------------------------------------------------------------------------
# ingest helpers (pure parts — no store needed)
# ---------------------------------------------------------------------------
class TestIngestHelpers:
    def test_normalize_signature_from_messages(self):
        from llm_mem0.ingest import _normalize_signature
        u, a, c = _normalize_signature(
            [
                {"role": "system", "content": "ctx"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ],
            None, None, "",
        )
        assert (u, a, c) == ("hi", "yo", "ctx")

    def test_normalize_signature_kwargs_win(self):
        from llm_mem0.ingest import _normalize_signature
        u, a, c = _normalize_signature(
            [{"role": "user", "content": "old"}], "new", None, "")
        assert u == "new"

    def test_build_base_meta_allowlist(self):
        from llm_mem0.ingest import _build_base_meta
        meta = _build_base_meta({
            "channel_id": "123",
            "importance": 5,          # trust-relevant → must be dropped
            "gate_version": "forged",  # trust-relevant → must be dropped
            "persona": "kanae",
        })
        assert meta["channel_id"] == "123"
        assert meta["persona"] == "kanae"
        assert meta["gate_version"] == "v2"
        assert "importance" not in meta

    def test_filter_candidates(self, monkeypatch):
        from llm_mem0.ingest import _filter_candidates
        facts = [
            {"speaker": "self", "importance": 3, "text": "A"},
            {"speaker": "other", "importance": 5, "text": "B"},
            {"speaker": "self", "importance": 1, "text": "C"},
            {"speaker": "self", "importance": 4, "text": "A"},  # intra-batch dup
        ]
        kept, low, other, dup = _filter_candidates(facts)
        assert [f["text"] for f in kept] == ["A"]
        assert (low, other, dup) == (1, 1, 1)

    def test_build_fact_meta_scalar_only(self):
        from llm_mem0.ingest import _build_fact_meta
        meta = _build_fact_meta(
            {
                "text": "体重58kg",
                "importance": 4,
                "category": "personal",
                "tags": ["health", "we,ight"],
                "attribute": "weight",
                "value": {"type": "quantity", "value": 58.0, "unit": "kg"},
                "condition": "",
                "scope": "conversation",
            },
            {"timestamp": 1, "gate_version": "v2", "channel_id": "99"},
        )
        assert meta["tags"] == "health,we ight"          # CSV, comma-scrubbed
        assert meta["attribute"] == "weight"
        assert json.loads(meta["value_json"])["unit"] == "kg"
        assert meta["scope"] == "conversation:99"
        assert all(v is not None for v in meta.values())
        assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())

    def test_build_fact_meta_low_importance_forces_warm_tier(self):
        from llm_mem0.ingest import _build_fact_meta
        meta = _build_fact_meta(
            {"text": "x", "importance": 2}, {"timestamp": 1, "gate_version": "v2"})
        assert meta["tier"] == 2

    def test_build_fact_meta_preserves_live_tier_zero(self):
        from llm_mem0.ingest import _build_fact_meta
        meta = _build_fact_meta(
            {"text": "x", "importance": 5, "tier": 0},
            {"timestamp": 1, "gate_version": "v2"})
        assert meta["tier"] == 0  # Live must not be demoted to Hot

    def test_dedup_single_threshold(self, monkeypatch):
        from llm_mem0.ingest import _dedup_single_threshold
        monkeypatch.setenv("MEM0_CLIENT_DEDUP_THRESHOLD", "0.2")
        candidates = [{"text": "a"}, {"text": "b"}]
        near = [
            [{"score": 0.1}],   # below threshold → dup
            [{"score": 0.9}],   # above → kept
        ]
        kept, dropped, reinforced, archive = _dedup_single_threshold(candidates, near)
        assert [f["text"] for f in kept] == ["b"]
        assert dropped == 1 and reinforced == [] and archive == []


# ---------------------------------------------------------------------------
# mem0ai API-generation compat (user_id kwarg vs filters={"user_id": ...})
# ---------------------------------------------------------------------------
class TestMem0ApiCompat:
    class _LegacySdk:
        def search(self, *, query, user_id, limit):
            return {"results": [{"memory": f"legacy:{user_id}"}]}

        def get_all(self, *, user_id, limit=None):
            return {"results": [{"memory": f"legacy:{user_id}"}]}

    class _NewSdk:
        def search(self, *, query, limit, user_id=None, filters=None):
            if user_id is not None:
                raise ValueError(
                    "Top-level entity parameters frozenset({'user_id'}) are "
                    "not supported in search(). Use filters={'user_id': '...'}.")
            return {"results": [{"memory": f"new:{filters['user_id']}"}]}

        def get_all(self, *, limit=None, user_id=None, filters=None):
            if user_id is not None:
                raise ValueError(
                    "Top-level entity parameters frozenset({'user_id'}) are "
                    "not supported in get_all(). Use filters={'user_id': '...'}.")
            return {"results": [{"memory": f"new:{filters['user_id']}"}]}

    @pytest.fixture(autouse=True)
    def _reset_style(self, monkeypatch):
        from llm_mem0 import client
        monkeypatch.setattr(client, "_filters_style", None)

    def test_legacy_sdk(self):
        from llm_mem0.client import mem_get_all, mem_search
        sdk = self._LegacySdk()
        assert mem_search(sdk, query="q", user_id="u", limit=3)["results"][0]["memory"] == "legacy:u"
        assert mem_get_all(sdk, user_id="u")["results"][0]["memory"] == "legacy:u"

    def test_new_sdk_switches_and_sticks(self):
        from llm_mem0 import client
        sdk = self._NewSdk()
        out = client.mem_search(sdk, query="q", user_id="u", limit=3)
        assert out["results"][0]["memory"] == "new:u"
        assert client._filters_style is True
        # get_all now goes straight to filters style (no retry needed)
        out = client.mem_get_all(sdk, user_id="u", limit=5)
        assert out["results"][0]["memory"] == "new:u"

    def test_unrelated_error_propagates(self):
        from llm_mem0.client import mem_search

        class Broken:
            def search(self, **kw):
                raise RuntimeError("connection refused")

        with pytest.raises(RuntimeError):
            mem_search(Broken(), query="q", user_id="u", limit=1)


# ---------------------------------------------------------------------------
# batch: lost-update guard
# ---------------------------------------------------------------------------
class TestBatch:
    def test_concurrent_append_no_lost_update(self, tmp_path):
        from llm_mem0 import batch
        n_threads, per_thread = 8, 10

        def worker():
            for _ in range(per_thread):
                batch.append_turn(tmp_path, 42, "u", "a")

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        turns, _ = batch.pop_batch(tmp_path, 42)
        assert len(turns) == n_threads * per_thread
        # pop resets
        turns2, _ = batch.pop_batch(tmp_path, 42)
        assert turns2 == []


# ---------------------------------------------------------------------------
# MCP tools (no ChromaDB — error paths must surface, not fake success)
# ---------------------------------------------------------------------------
class TestMcpTools:
    @pytest.fixture(autouse=True)
    def _no_chroma(self, monkeypatch, tmp_path):
        pytest.importorskip("mcp")
        monkeypatch.setenv("LLM_MEM0_STATE_DIR", str(tmp_path))
        monkeypatch.setenv("CHROMA_PORT", "59999")  # nothing listens here
        monkeypatch.delenv("MEM0_DEFAULT_USER_ID", raising=False)
        # reset client singleton state so each test gets the init-failure path
        from llm_mem0 import client
        monkeypatch.setattr(client, "_mem0_instance", None)
        monkeypatch.setattr(client, "_init_failed_at", None)

    @pytest.mark.asyncio
    async def test_registered_tools(self):
        from llm_mem0 import mcp_server
        tools = await mcp_server.mcp.list_tools()
        assert sorted(t.name for t in tools) == [
            "add_memory", "delete_memory", "list_memories", "search_memory",
        ]

    @pytest.mark.asyncio
    async def test_user_id_required_without_default(self):
        from llm_mem0 import mcp_server
        out = await mcp_server.add_memory("hello")
        assert out.startswith("Error: user_id is required")

    @pytest.mark.asyncio
    async def test_default_user_id_env(self, monkeypatch):
        from llm_mem0 import mcp_server
        monkeypatch.setenv("MEM0_DEFAULT_USER_ID", "u1")
        out = await mcp_server.add_memory("hello")
        # store is down — the tool must report it, not claim success
        assert out.startswith("Error: memory store unavailable")

    @pytest.mark.asyncio
    async def test_add_memory_surfaces_pipeline_error(self, monkeypatch):
        from llm_mem0 import mcp_server

        async def fake_add(**kw):
            return {"error": "ValueError: boom", "results": [], "kept": 0,
                    "skipped": {}}

        monkeypatch.setattr(mcp_server, "add_memories", fake_add)
        out = await mcp_server.add_memory("x", user_id="u1")
        assert out == "Error: ValueError: boom"

    @pytest.mark.asyncio
    async def test_delete_requires_id(self):
        from llm_mem0 import mcp_server
        assert (await mcp_server.delete_memory("")) == "Error: memory_id is required."
