"""Tests for Phase ⑦: feeling-of-knowing gate + calibrated null-result
sentinel, and the recall-reinforcement firing hook.

Ported from the Discord bot's characterization suite.
"""

from __future__ import annotations

import asyncio

import pytest

from llm_mem0.format import format_memories_for_prompt
from llm_mem0.search import (
    _finalize_smart_result,
    assess_recall_confidence,
)


class TestAssessRecallConfidence:
    """Phase ⑦: feeling-of-knowing gate — pure scoring logic."""

    def test_confident_when_best_distance_is_close(self):
        result = assess_recall_confidence([{"score": 0.1}, {"score": 0.5}])
        assert result["status"] == "confident"
        assert result["best_distance"] == 0.1
        assert result["candidate_count"] == 2

    def test_weak_when_best_distance_is_middling(self):
        result = assess_recall_confidence([{"score": 0.9}])
        assert result["status"] == "weak"

    def test_no_memory_when_best_distance_exceeds_threshold(self):
        result = assess_recall_confidence([{"score": 1.3}])
        assert result["status"] == "no_memory"

    def test_no_memory_when_candidates_empty(self):
        result = assess_recall_confidence([])
        assert result["status"] == "no_memory"
        assert result["best_distance"] is None
        assert result["candidate_count"] == 0

    def test_no_memory_when_all_scores_missing(self):
        result = assess_recall_confidence([{"id": "f1"}, {"id": "f2"}])
        assert result["status"] == "no_memory"
        assert result["best_distance"] is None

    def test_ignores_non_numeric_scores(self):
        result = assess_recall_confidence([{"score": None}, {"score": 0.2}])
        assert result["status"] == "confident"
        assert result["best_distance"] == 0.2

    def test_thresholds_are_dynamic_settings(self, monkeypatch):
        monkeypatch.setenv("MEM0_FOK_WEAK_DISTANCE", "0.3")
        result = assess_recall_confidence([{"score": 0.5}])
        assert result["status"] == "weak"


class TestFinalizeSmartResult:
    """Phase ⑦: core-memory merge + sentinel injection."""

    @pytest.mark.asyncio
    async def test_no_fok_block_returns_plain_results(self):
        results = [{"id": "f1", "memory": "text"}]
        out = await _finalize_smart_result(
            results, None, 5, {"best_distance": 0.1}, fok_block=False,
        )
        assert out == results

    @pytest.mark.asyncio
    async def test_fok_block_appends_sentinel(self):
        out = await _finalize_smart_result(
            [], None, 5, {"best_distance": 1.4}, fok_block=True,
        )
        assert len(out) == 1
        assert out[0]["_fok_no_memory"] is True
        assert out[0]["best_distance"] == 1.4

    @pytest.mark.asyncio
    async def test_fok_block_still_includes_core_memories(self):
        async def _core_task():
            return [{"id": "core1", "memory": "coreことば", "_core": True}]

        task = asyncio.ensure_future(_core_task())
        out = await _finalize_smart_result(
            [], task, 5, {"best_distance": 1.4}, fok_block=True,
        )
        ids = {m.get("id") for m in out}
        assert "core1" in ids
        assert any(m.get("_fok_no_memory") for m in out)


class TestFokSentinelRendering:
    """format_memories_for_prompt renders the sentinel as a calibrated
    null result instead of dropping it."""

    def test_sentinel_only_renders_null_result_line(self):
        block = format_memories_for_prompt(
            [{"_fok_no_memory": True, "best_distance": 1.3}])
        assert "一致する具体的な記憶は見つからなかった" in block
        assert "[Long-term memory" in block

    def test_sentinel_alongside_core_memories(self):
        block = format_memories_for_prompt([
            {"memory": "コア事実", "_core": True},
            {"_fok_no_memory": True, "best_distance": 1.3},
        ])
        assert "コア事実" in block
        assert "一致する具体的な記憶は見つからなかった" in block

    def test_no_sentinel_no_null_line(self):
        block = format_memories_for_prompt([{"memory": "普通の事実"}])
        assert "見つからなかった" not in block


class TestFireRecallReinforcement:
    """Recall-reinforcement firing: gated by settings, excludes core /
    episode-bundle rows, fire-and-forget."""

    @pytest.mark.asyncio
    async def test_disabled_by_default_no_task(self, monkeypatch):
        from llm_mem0 import search

        called = []
        monkeypatch.setattr(
            search, "_get_mem0", lambda: called.append("get") or None)
        search._fire_recall_reinforcement([{"id": "f1"}])
        await asyncio.sleep(0.05)
        assert called == []  # gate off → nothing spawned

    @pytest.mark.asyncio
    async def test_fires_for_real_hits_only(self, monkeypatch):
        from llm_mem0 import search
        from llm_mem0 import reinforce as reinforce_mod

        monkeypatch.setenv("MEM0_RECALL_REINFORCE_ENABLED", "1")
        reinforced: list[str] = []

        class FakeMem:
            pass

        async def fake_reinforce(mem, mid):
            reinforced.append(mid)
            return True

        monkeypatch.setattr(search, "_get_mem0", lambda: FakeMem())
        monkeypatch.setattr(reinforce_mod, "reinforce_on_recall", fake_reinforce)
        search._fire_recall_reinforcement([
            {"id": "f1"},
            {"id": "core1", "_core": True},
            {"id": "b1", "_episode_bundle": True},
            {"_fok_no_memory": True},
        ])
        await asyncio.sleep(0.1)
        assert reinforced == ["f1"]
