"""Tests for Phase ④: session priming buffer, degree-normalized entity
co-occurrence, and multi-hop spreading-activation graph expansion.

Ported from the Discord bot's characterization suite; TTL/eviction knobs
are exercised through env vars because llm_mem0 settings are dynamic.
"""

from __future__ import annotations

import time as _time
from unittest.mock import MagicMock

import pytest

from llm_mem0 import graph as graph_mod
from llm_mem0 import priming


@pytest.fixture(autouse=True)
def _clear_priming_buffer():
    priming.clear()
    yield
    priming.clear()


class TestTouchAndGetPrimedEntities:
    def test_touch_then_get(self):
        priming.touch("u1", "ch1", [10, 20])
        assert priming.get_primed_entities("u1", "ch1") == {10, 20}

    def test_scoped_per_channel(self):
        priming.touch("u1", "ch1", [10])
        priming.touch("u1", "ch2", [99])
        assert priming.get_primed_entities("u1", "ch1") == {10}
        assert priming.get_primed_entities("u1", "ch2") == {99}

    def test_scoped_per_user(self):
        priming.touch("u1", "ch1", [10])
        priming.touch("u2", "ch1", [10])
        priming.clear("u2", "ch1")
        assert priming.get_primed_entities("u1", "ch1") == {10}
        assert priming.get_primed_entities("u2", "ch1") == set()

    def test_ignores_empty_input(self):
        priming.touch("u1", "ch1", [])
        priming.touch("", "ch1", [10])
        assert priming.get_primed_entities("u1", "ch1") == set()


class TestTtlAndEviction:
    def test_entries_expire_after_ttl(self, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr(priming.time, "monotonic", lambda: clock["t"])
        monkeypatch.setenv("MEM0_PRIMING_TTL_SEC", "60")

        priming.touch("u1", "ch1", [10])
        clock["t"] += 30
        assert priming.get_primed_entities("u1", "ch1") == {10}
        clock["t"] += 40  # total 70s elapsed > 60s TTL
        assert priming.get_primed_entities("u1", "ch1") == set()

    def test_max_entities_evicts_stalest_first(self, monkeypatch):
        clock = {"t": 0.0}
        monkeypatch.setattr(priming.time, "monotonic", lambda: clock["t"])
        monkeypatch.setenv("MEM0_PRIMING_MAX_ENTITIES", "3")

        for i in range(5):
            clock["t"] += 1
            priming.touch("u1", "ch1", [i + 1])  # 1..5 (0 is falsy, skipped)

        live = priming.get_primed_entities("u1", "ch1")
        assert live == {3, 4, 5}  # the 3 most-recently-touched survive


class TestClear:
    def test_clear_all(self):
        priming.touch("u1", "ch1", [1])
        priming.touch("u2", "ch1", [2])
        priming.clear()
        assert priming.get_primed_entities("u1", "ch1") == set()
        assert priming.get_primed_entities("u2", "ch1") == set()


class TestCoOccurringEntities:
    """Phase ④: degree-normalized co-occurrence weights (spreading activation)."""

    @pytest.fixture(autouse=True)
    def _tmp_db(self, tmp_path):
        graph_mod._reset_db_path_for_test(str(tmp_path / "graph.sqlite"))
        graph_mod.invalidate_top_entities_cache()
        yield
        graph_mod._reset_db_path_for_test(None)
        graph_mod.invalidate_top_entities_cache()

    def _entity(self, name, etype="media"):
        return graph_mod.upsert_entity(name, etype, "u1")

    def test_entity_degree_counts_linked_facts(self):
        a = self._entity("Rust")
        graph_mod.add_edge("f1", a)
        graph_mod.add_edge("f2", a)
        assert graph_mod.get_entity_degree(a) == 2

    def test_entity_degree_zero_for_unlinked_entity(self):
        a = self._entity("Unlinked")
        assert graph_mod.get_entity_degree(a) == 0

    def test_finds_shared_fact_partners(self):
        rust = self._entity("Rust")
        async_ = self._entity("async")
        unrelated = self._entity("Unrelated")
        graph_mod.add_edge("f1", rust)
        graph_mod.add_edge("f1", async_)  # f1 mentions both -> co-occurrence
        graph_mod.add_edge("f2", unrelated)

        results = graph_mod.get_co_occurring_entities(rust)
        ids = [r[0] for r in results]
        assert async_ in ids
        assert unrelated not in ids

    def test_weight_normalized_by_degree(self):
        """A high-degree hub entity should NOT dominate purely from size —
        its normalized weight to a low-degree partner should be lower than a
        symmetric pair with equal degree."""
        hub = self._entity("Hub")
        partner = self._entity("Partner")
        for i in range(10):
            graph_mod.add_edge(f"hub_only_{i}", hub)  # inflate hub's degree
        graph_mod.add_edge("shared1", hub)
        graph_mod.add_edge("shared1", partner)  # single shared fact

        sym_a = self._entity("SymA")
        sym_b = self._entity("SymB")
        graph_mod.add_edge("sym_shared", sym_a)
        graph_mod.add_edge("sym_shared", sym_b)  # equal, low degree (1 each)

        hub_weight = dict(graph_mod.get_co_occurring_entities(hub)).get(partner)
        sym_weight = dict(graph_mod.get_co_occurring_entities(sym_a)).get(sym_b)

        assert hub_weight is not None and sym_weight is not None
        assert hub_weight < sym_weight

    def test_empty_for_isolated_entity(self):
        solo = self._entity("Solo")
        graph_mod.add_edge("f1", solo)
        assert graph_mod.get_co_occurring_entities(solo) == []

    def test_respects_limit(self):
        center = self._entity("Center")
        for i in range(5):
            other = self._entity(f"Other{i}")
            graph_mod.add_edge("shared_fact", center)
            graph_mod.add_edge("shared_fact", other)
        results = graph_mod.get_co_occurring_entities(center, limit=2)
        assert len(results) <= 2


class TestGraphExpandCandidates:
    """Phase ④: multi-hop spreading-activation graph expansion, with
    priming buffer integration."""

    def _wire_graph(self, monkeypatch, *, fact_entities, co_occurring, related_facts):
        entity_cls = type("E", (), {})

        def _entity(eid):
            e = entity_cls()
            e.id = eid
            return e

        monkeypatch.setattr(
            "llm_mem0.graph.get_fact_entities",
            lambda fid: [_entity(eid) for eid in fact_entities.get(fid, [])],
        )
        monkeypatch.setattr(
            "llm_mem0.graph.get_co_occurring_entities",
            lambda eid, limit=5: co_occurring.get(eid, []),
        )
        monkeypatch.setattr(
            "llm_mem0.graph.get_related_facts",
            lambda eid, limit=5, exclude_fact_ids=(): [
                f for f in related_facts.get(eid, []) if f not in set(exclude_fact_ids)
            ],
        )

    def _fake_mem(self, facts: dict):
        mem = MagicMock()
        facts_with_id = {fid: {"id": fid, **body} for fid, body in facts.items()}
        mem.get.side_effect = lambda fid: facts_with_id.get(fid)
        return mem

    @pytest.mark.asyncio
    async def test_no_candidates_returns_immediately(self):
        from llm_mem0.search import _graph_expand_candidates
        assert await _graph_expand_candidates([], "u1") == []

    @pytest.mark.asyncio
    async def test_spreads_activation_across_two_hops_and_scores_by_distance(self, monkeypatch):
        monkeypatch.setattr(
            "llm_mem0.search._get_mem0",
            lambda: self._fake_mem({
                "s1": {"memory": "sibling1", "metadata": {}},
                "s2": {"memory": "sibling2", "metadata": {}},
            }),
        )
        self._wire_graph(
            monkeypatch,
            fact_entities={"c1": [1]},
            co_occurring={1: [(2, 0.8)], 2: []},
            related_facts={1: ["s1"], 2: ["s2"]},
        )
        from llm_mem0.search import _graph_expand_candidates
        from llm_mem0.settings import settings

        candidates = [{"id": "c1", "score": 0.1}]
        result = await _graph_expand_candidates(candidates, "u1", current_channel_id=None)

        by_id = {r["id"]: r for r in result if r.get("_graph_expanded")}
        assert set(by_id) == {"s1", "s2"}
        # s1 reached via the seed entity directly (hop 0) -> higher
        # activation -> lower (closer) distance than s2 (hop-1 decay).
        assert by_id["s1"]["score"] < by_id["s2"]["score"]
        assert by_id["s1"]["_graph_activation"] > by_id["s2"]["_graph_activation"]
        assert by_id["s1"]["score"] >= settings.MEM0_GRAPH_SIBLING_MIN_DISTANCE

    @pytest.mark.asyncio
    async def test_seed_entities_are_recorded_in_priming_buffer(self, monkeypatch):
        monkeypatch.setattr(
            "llm_mem0.search._get_mem0",
            lambda: self._fake_mem({"s1": {"memory": "sibling1", "metadata": {}}}),
        )
        self._wire_graph(
            monkeypatch,
            fact_entities={"c1": [42]},
            co_occurring={42: []},
            related_facts={42: ["s1"]},
        )
        from llm_mem0.search import _graph_expand_candidates
        candidates = [{"id": "c1", "score": 0.2}]
        await _graph_expand_candidates(candidates, "u1", current_channel_id="ch1")

        assert priming.get_primed_entities("u1", "ch1") == {42}

    @pytest.mark.asyncio
    async def test_primed_entities_get_activation_top_up(self, monkeypatch):
        monkeypatch.setattr(
            "llm_mem0.search._get_mem0",
            lambda: self._fake_mem({"s1": {"memory": "sibling1", "metadata": {}}}),
        )
        self._wire_graph(
            monkeypatch,
            fact_entities={"c1": [1]},
            co_occurring={1: [(2, 0.5)]},
            related_facts={2: ["s1"]},
        )
        from llm_mem0.search import _graph_expand_candidates
        candidates = [{"id": "c1", "score": 0.5}]

        priming.clear()
        baseline = await _graph_expand_candidates(list(candidates), "u1", current_channel_id="ch1")
        baseline_score = next(r["score"] for r in baseline if r["id"] == "s1")

        priming.clear()
        priming.touch("u1", "ch1", [2])  # entity 2 was primed by a prior turn
        boosted = await _graph_expand_candidates(list(candidates), "u1", current_channel_id="ch1")
        boosted_score = next(r["score"] for r in boosted if r["id"] == "s1")

        assert boosted_score < baseline_score  # priming makes the distance smaller

    @pytest.mark.asyncio
    async def test_archived_sibling_is_filtered_out(self, monkeypatch):
        monkeypatch.setattr(
            "llm_mem0.search._get_mem0",
            lambda: self._fake_mem({
                "s1": {"memory": "old fact",
                       "metadata": {"valid_to": int(_time.time()) - 10}},
            }),
        )
        self._wire_graph(
            monkeypatch,
            fact_entities={"c1": [1]},
            co_occurring={1: []},
            related_facts={1: ["s1"]},
        )
        from llm_mem0.search import _graph_expand_candidates
        candidates = [{"id": "c1", "score": 0.2}]
        result = await _graph_expand_candidates(candidates, "u1")
        assert result == candidates  # archived sibling excluded
