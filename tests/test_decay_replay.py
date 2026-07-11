"""Tests for Phase ② (decay: tier decay + gist compression) and Phase ③
(replay + schema abstraction). Ported from the Discord bot's suite; the
bot-side oauth-refresh patches are dropped (token freshness is the auth
backend's concern in llm_mem0).
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_mem0 import decay, replay


@pytest.fixture
def tmp_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MEM0_STATE_DIR", str(tmp_path))
    return tmp_path


class TestActivationToTier:
    def test_live(self):
        assert decay.activation_to_tier(-1.0) == 0

    def test_hot(self):
        assert decay.activation_to_tier(-7.0) == 1

    def test_warm(self):
        assert decay.activation_to_tier(-8.0) == 2

    def test_cold(self):
        assert decay.activation_to_tier(-10.0) == 3

    def test_neg_inf_is_cold(self):
        assert decay.activation_to_tier(float("-inf")) == 3


class TestIsGistExempt:
    def test_high_importance_exempt(self):
        assert decay.is_gist_exempt({"importance": 5}) is True

    def test_low_importance_not_exempt(self):
        assert decay.is_gist_exempt({"importance": 2}) is False

    def test_attribute_tagged_exempt(self):
        assert decay.is_gist_exempt({"importance": 2, "attribute": "residence"}) is True

    def test_already_gist_exempt(self):
        assert decay.is_gist_exempt({"importance": 2, "gist_of": "a,b,c"}) is True


class TestRecomputeTiersForUser:
    @pytest.mark.asyncio
    async def test_writes_changed_tiers_only(self, monkeypatch):
        now = time.time()
        rows = [
            {  # stale — was tier=0 (Live) but hasn't been touched in 2 years
                "id": "f1", "memory": "古い話",
                "metadata": {"tier": 0, "timestamp": int(now - 86400 * 730), "mention_count": 1},
            },
            {  # freshly reinforced — already correctly tier=0
                "id": "f2", "memory": "最近の話",
                "metadata": {"tier": 0, "timestamp": int(now - 60), "mention_count": 1},
            },
            {  # archived — must be skipped entirely
                "id": "f3", "memory": "アーカイブ済み",
                "metadata": {"tier": 0, "timestamp": int(now - 86400 * 365), "valid_to": int(now - 1)},
            },
        ]
        mem = MagicMock()
        monkeypatch.setattr("llm_mem0.client._get_mem0", lambda: mem)
        monkeypatch.setattr(
            "llm_mem0.client.get_all_memories", AsyncMock(return_value=rows),
        )

        summary = await decay.recompute_tiers_for_user("u1", dry_run=False)

        assert summary["checked"] == 2  # f3 excluded (archived)
        assert summary["changed"] == 1  # only f1 transitions
        mem.update.assert_called_once()
        assert mem.update.call_args.kwargs["memory_id"] == "f1"
        assert mem.update.call_args.kwargs["metadata"]["tier"] == 3

    @pytest.mark.asyncio
    async def test_dry_run_never_writes(self, monkeypatch):
        now = time.time()
        rows = [{
            "id": "f1", "memory": "古い話",
            "metadata": {"tier": 0, "timestamp": int(now - 86400 * 730), "mention_count": 1},
        }]
        mem = MagicMock()
        monkeypatch.setattr("llm_mem0.client._get_mem0", lambda: mem)
        monkeypatch.setattr(
            "llm_mem0.client.get_all_memories", AsyncMock(return_value=rows),
        )

        summary = await decay.recompute_tiers_for_user("u1", dry_run=True)

        assert summary["changed"] == 1  # "would change"
        mem.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_mem_returns_empty_summary(self, monkeypatch):
        monkeypatch.setattr("llm_mem0.client._get_mem0", lambda: None)
        summary = await decay.recompute_tiers_for_user("u1", dry_run=True)
        assert summary == {"checked": 0, "changed": 0, "transitions": {}}


class TestGistCompressForUser:
    @pytest.mark.asyncio
    async def test_clusters_by_entity_and_archives_originals(self, monkeypatch):
        rows = [
            {"id": f"f{i}", "memory": f"Rust の話 {i}",
             "metadata": {"tier": 3, "importance": 2, "category": "tech"}}
            for i in range(3)
        ] + [
            {"id": "exempt1", "memory": "体重58kg",
             "metadata": {"tier": 3, "importance": 5}},  # exempt: imp>3
        ]
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "gist1"}]}
        monkeypatch.setattr("llm_mem0.client._get_mem0", lambda: mem)
        monkeypatch.setattr(
            "llm_mem0.client.get_all_memories", AsyncMock(return_value=rows),
        )

        entity = MagicMock(id=1, mention_count=5)

        def _fake_get_fact_entities(fact_id):
            return [entity] if fact_id.startswith("f") else []

        monkeypatch.setattr("llm_mem0.graph.get_fact_entities", _fake_get_fact_entities)
        monkeypatch.setattr("llm_mem0.graph.add_edge", MagicMock(return_value=True))

        async def _fake_synthesize_cluster(texts):
            assert len(texts) == 3
            return ["Rust の話をまとめて3回した"]

        monkeypatch.setattr(
            "llm_mem0.dedup.synthesize_cluster", _fake_synthesize_cluster,
        )

        summary = await decay.gist_compress_for_user("u1", dry_run=False)

        assert summary["clusters"] == 1
        assert summary["compressed"] == 1
        assert summary["archived"] == 3
        archived_ids = {c.kwargs["memory_id"] for c in mem.update.call_args_list}
        assert archived_ids == {"f0", "f1", "f2"}
        for c in mem.update.call_args_list:
            assert c.kwargs["metadata"]["valid_to"]

    @pytest.mark.asyncio
    async def test_dry_run_never_writes(self, monkeypatch):
        rows = [
            {"id": f"f{i}", "memory": f"Rust の話 {i}",
             "metadata": {"tier": 3, "importance": 2}}
            for i in range(3)
        ]
        mem = MagicMock()
        monkeypatch.setattr("llm_mem0.client._get_mem0", lambda: mem)
        monkeypatch.setattr(
            "llm_mem0.client.get_all_memories", AsyncMock(return_value=rows),
        )
        entity = MagicMock(id=1, mention_count=5)
        monkeypatch.setattr(
            "llm_mem0.graph.get_fact_entities", lambda fid: [entity],
        )

        async def _fake_synthesize_cluster(texts):
            return ["まとめ"]

        monkeypatch.setattr(
            "llm_mem0.dedup.synthesize_cluster", _fake_synthesize_cluster,
        )

        summary = await decay.gist_compress_for_user("u1", dry_run=True)

        assert summary["clusters"] == 1
        mem.update.assert_not_called()
        mem.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_mem_returns_empty_summary(self, monkeypatch):
        monkeypatch.setattr("llm_mem0.client._get_mem0", lambda: None)
        summary = await decay.gist_compress_for_user("u1", dry_run=True)
        assert summary == {"clusters": 0, "compressed": 0, "archived": 0}


# ---------------------------------------------------------------------------
# Phase ③ replay
# ---------------------------------------------------------------------------
class TestPairTurns:
    def test_pairs_user_with_following_assistant(self):
        lines = [
            {"ts": "t1", "role": "user", "author": "", "text": "こんにちは"},
            {"ts": "t2", "role": "assistant", "author": "", "text": "やあ"},
        ]
        turns, last_ts = replay._pair_turns(lines)
        assert turns == [("こんにちは", "やあ")]
        assert last_ts == "t2"

    def test_user_without_assistant_reply(self):
        lines = [{"ts": "t1", "role": "user", "author": "", "text": "誰か見てる？"}]
        turns, last_ts = replay._pair_turns(lines)
        assert turns == [("誰か見てる？", "")]
        assert last_ts == "t1"

    def test_skips_channel_role_lines(self):
        lines = [
            {"ts": "t1", "role": "channel", "author": "someone", "text": "雑談"},
            {"ts": "t2", "role": "user", "author": "", "text": "本題"},
            {"ts": "t3", "role": "assistant", "author": "", "text": "了解"},
        ]
        turns, last_ts = replay._pair_turns(lines)
        assert turns == [("本題", "了解")]
        assert last_ts == "t3"

    def test_empty_input(self):
        assert replay._pair_turns([]) == ([], "")


class TestReplayChannelHistory:
    @pytest.mark.asyncio
    async def test_dry_run_never_advances_cursor(self, monkeypatch, tmp_state_dir):
        lines = [
            {"ts": "t1", "role": "user", "author": "", "text": "Rust勉強してる"},
            {"ts": "t2", "role": "assistant", "author": "", "text": "いいね"},
        ]
        monkeypatch.setattr(
            "llm_mem0.history_index.get_lines_since",
            lambda ch, since, limit: lines,
        )
        add_memories_mock = AsyncMock(return_value={"kept": 1, "results": []})
        monkeypatch.setattr("llm_mem0.ingest.add_memories", add_memories_mock)

        summary = await replay.replay_channel_history(123, "u1", dry_run=True)

        assert summary["turns_processed"] == 1
        assert summary["facts_kept"] == 0  # dry-run never calls add_memories
        add_memories_mock.assert_not_called()
        assert not replay._cursor_path().exists()

    @pytest.mark.asyncio
    async def test_apply_advances_cursor_and_tags_source(self, monkeypatch, tmp_state_dir):
        lines = [
            {"ts": "t1", "role": "user", "author": "", "text": "Rust勉強してる"},
            {"ts": "t2", "role": "assistant", "author": "", "text": "いいね"},
        ]
        monkeypatch.setattr(
            "llm_mem0.history_index.get_lines_since",
            lambda ch, since, limit: lines,
        )
        add_memories_mock = AsyncMock(return_value={"kept": 2, "results": []})
        monkeypatch.setattr("llm_mem0.ingest.add_memories", add_memories_mock)

        summary = await replay.replay_channel_history(123, "u1", dry_run=False)

        assert summary["turns_processed"] == 1
        assert summary["facts_kept"] == 2
        assert summary["cursor_advanced_to"] == "t2"
        add_memories_mock.assert_called_once()
        kwargs = add_memories_mock.call_args.kwargs
        assert kwargs["metadata"]["source"] == "replay"
        assert kwargs["user_text"] == "Rust勉強してる"
        assert kwargs["assistant_text"] == "いいね"

        cursor = replay._load_cursor()
        assert cursor["123"] == "t2"

    @pytest.mark.asyncio
    async def test_second_run_starts_from_saved_cursor(self, monkeypatch, tmp_state_dir):
        seen_since = []

        def _fake_get_lines_since(ch, since, limit):
            seen_since.append(since)
            return []

        monkeypatch.setattr(
            "llm_mem0.history_index.get_lines_since", _fake_get_lines_since,
        )
        monkeypatch.setattr(
            "llm_mem0.ingest.add_memories",
            AsyncMock(return_value={"kept": 0, "results": []}),
        )

        replay._save_cursor({"123": "t5"})
        await replay.replay_channel_history(123, "u1", dry_run=False)

        assert seen_since == ["t5"]


def _entity(eid, etype, mention_count=5):
    e = MagicMock()
    e.id = eid
    e.type = etype
    e.mention_count = mention_count
    return e


class TestAbstractSchemasForUser:
    @pytest.mark.asyncio
    async def test_requires_distinct_entities(self, monkeypatch):
        rows = [
            {"id": "f1", "memory": "Blade Runnerが好き",
             "metadata": {"category": "hobby", "subcategory": "movie"}},
            {"id": "f2", "memory": "伊藤計劃が好き",
             "metadata": {"category": "hobby", "subcategory": "movie"}},
        ]  # only 2 distinct entities — below MEM0_SCHEMA_MIN_DISTINCT_ENTITIES (3)
        monkeypatch.setattr(
            "llm_mem0.client.get_all_memories", AsyncMock(return_value=rows),
        )

        def _fake_entities(fid):
            return [_entity({"f1": 1, "f2": 2}[fid], "media")]

        monkeypatch.setattr("llm_mem0.graph.get_fact_entities", _fake_entities)
        add_memories_mock = AsyncMock()
        monkeypatch.setattr("llm_mem0.ingest.add_memories", add_memories_mock)

        summary = await replay.abstract_schemas_for_user("u1", dry_run=False)

        assert summary["groups_checked"] == 0
        assert summary["schemas_created"] == 0
        add_memories_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_generates_and_inserts_with_provenance(self, monkeypatch):
        rows = [
            {"id": f"f{i}", "memory": f"作品{i}が好き",
             "metadata": {"category": "hobby", "subcategory": "movie"}}
            for i in range(3)
        ]
        monkeypatch.setattr(
            "llm_mem0.client.get_all_memories", AsyncMock(return_value=rows),
        )

        def _fake_entities(fid):
            return [_entity(int(fid[1:]), "media")]

        monkeypatch.setattr("llm_mem0.graph.get_fact_entities", _fake_entities)

        async def _fake_generate(texts):
            assert len(texts) == 3
            return ["SF作品が好き"]

        monkeypatch.setattr(replay, "_generate_schema_facts", _fake_generate)
        add_memories_mock = AsyncMock(return_value={"kept": 1, "results": []})
        monkeypatch.setattr("llm_mem0.ingest.add_memories", add_memories_mock)

        summary = await replay.abstract_schemas_for_user("u1", dry_run=False)

        assert summary["groups_checked"] == 1
        assert summary["schemas_created"] == 1
        add_memories_mock.assert_called_once()
        kwargs = add_memories_mock.call_args.kwargs
        fact = kwargs["pre_extracted_facts"][0]
        assert fact["text"] == "SF作品が好き"
        assert fact["tags"] == ["schema"]
        assert kwargs["metadata"]["source"] == "consolidation_schema"
        assert set(kwargs["metadata"]["derived_from"].split(",")) == {"f0", "f1", "f2"}

    @pytest.mark.asyncio
    async def test_dry_run_never_calls_add_memories(self, monkeypatch):
        rows = [
            {"id": f"f{i}", "memory": f"作品{i}が好き",
             "metadata": {"category": "hobby", "subcategory": "movie"}}
            for i in range(3)
        ]
        monkeypatch.setattr(
            "llm_mem0.client.get_all_memories", AsyncMock(return_value=rows),
        )

        def _fake_entities(fid):
            return [_entity(int(fid[1:]), "media")]

        monkeypatch.setattr("llm_mem0.graph.get_fact_entities", _fake_entities)

        async def _fake_generate(texts):
            return ["SF作品が好き"]

        monkeypatch.setattr(replay, "_generate_schema_facts", _fake_generate)
        add_memories_mock = AsyncMock()
        monkeypatch.setattr("llm_mem0.ingest.add_memories", add_memories_mock)

        summary = await replay.abstract_schemas_for_user("u1", dry_run=True)

        assert summary["schemas_created"] == 1
        add_memories_mock.assert_not_called()
