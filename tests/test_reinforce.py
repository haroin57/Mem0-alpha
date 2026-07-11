"""Tests for llm_mem0.reinforce — Hebbian dedup increments, recall
reinforcement (testing effect), and ACT-R base-level activation.

Ported from the Discord bot's characterization suite so the library keeps
the exact semantics the bot's production data was shaped by — notably the
importance-untouched contract (reinforcement must never feed back into the
LLM-assigned importance score).
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock

import pytest

from llm_mem0.reinforce import (
    MAX_REINFORCED_HISTORY,
    reinforce_existing_sync,
    reinforce_on_recall_sync,
    reinforcement_boost,
    _actr_activation,
    _actr_reinforcement_boost,
)


class TestReinforceExistingSync:
    def test_first_reinforce_increments_to_two(self):
        mem = MagicMock()
        mem.get.return_value = {
            "id": "m1",
            "memory": "先輩は Rust が好き",
            "metadata": {"importance": 3, "category": "tech"},
        }
        ok = reinforce_existing_sync(mem, "m1")
        assert ok is True
        mem.update.assert_called_once()
        kwargs = mem.update.call_args.kwargs
        partial = kwargs["metadata"]
        assert kwargs["memory_id"] == "m1"
        assert kwargs["data"] == "先輩は Rust が好き"
        # Partial update — only changed keys are sent
        assert partial["mention_count"] == 2
        # Bugfix contract: importance must NOT be touched by reinforcement
        # (it used to bump +1 per reinforcement and inflate trivial facts
        # into importance=5 "core memory" status).
        assert "importance" not in partial
        assert len(partial["reinforced_at"]) == 1
        assert partial["last_reinforced_at"] == partial["reinforced_at"][0]
        assert "category" not in partial  # preserved server-side

    def test_subsequent_reinforce(self):
        mem = MagicMock()
        mem.get.return_value = {
            "id": "m1",
            "memory": "fact text",
            "metadata": {
                "mention_count": 5,
                "importance": 4,
                "reinforced_at": [100, 200, 300],
                "last_reinforced_at": 300,
            },
        }
        reinforce_existing_sync(mem, "m1")
        partial = mem.update.call_args.kwargs["metadata"]
        assert partial["mention_count"] == 6
        assert "importance" not in partial
        assert len(partial["reinforced_at"]) == 4
        assert partial["last_reinforced_at"] > 300

    def test_importance_untouched_even_at_five(self):
        # Regression guard for the old "capped at 5" bump behavior: even a
        # fact already at importance=5 must not have reinforce rewrite
        # importance at all (not bump, not re-affirm — just leave it alone).
        mem = MagicMock()
        mem.get.return_value = {
            "id": "m1", "memory": "x", "metadata": {"importance": 5},
        }
        reinforce_existing_sync(mem, "m1")
        assert "importance" not in mem.update.call_args.kwargs["metadata"]

    def test_reinforced_at_trims_to_max_history(self):
        existing = list(range(20))  # 20 entries already
        mem = MagicMock()
        mem.get.return_value = {
            "id": "m1",
            "memory": "x",
            "metadata": {"mention_count": 21, "reinforced_at": existing},
        }
        reinforce_existing_sync(mem, "m1")
        partial = mem.update.call_args.kwargs["metadata"]
        assert len(partial["reinforced_at"]) == MAX_REINFORCED_HISTORY
        assert partial["reinforced_at"][-1] == partial["last_reinforced_at"]

    def test_empty_memory_id_returns_false(self):
        mem = MagicMock()
        assert reinforce_existing_sync(mem, "") is False
        mem.update.assert_not_called()

    def test_get_raising_is_fail_open(self):
        mem = MagicMock()
        mem.get.side_effect = RuntimeError("backend down")
        assert reinforce_existing_sync(mem, "m1") is False
        mem.update.assert_not_called()

    def test_update_raising_is_fail_open(self):
        mem = MagicMock()
        mem.get.return_value = {"id": "m1", "memory": "x", "metadata": {}}
        mem.update.side_effect = RuntimeError("write failed")
        assert reinforce_existing_sync(mem, "m1") is False

    def test_missing_memory_text_returns_false(self):
        # Without memory text we can't re-pass `data` to mem.update.
        mem = MagicMock()
        mem.get.return_value = {"id": "m1", "metadata": {}}
        assert reinforce_existing_sync(mem, "m1") is False
        mem.update.assert_not_called()


class TestReinforcementBoost:
    def test_none_metadata_returns_zero(self):
        assert reinforcement_boost(None) == 0.0

    def test_default_mention_one_no_freq_boost(self):
        # mention_count=1 → log(1 + α*0) = 0
        boost = reinforcement_boost({"mention_count": 1})
        assert boost == 0.0

    def test_higher_mention_means_higher_boost(self):
        low = reinforcement_boost({"mention_count": 2})
        high = reinforcement_boost({"mention_count": 20})
        assert high > low > 0.0

    def test_log_dampening(self):
        # 20 mentions shouldn't be 10x 2 mentions — log dampens
        b2 = reinforcement_boost({"mention_count": 2})
        b20 = reinforcement_boost({"mention_count": 20})
        assert b20 < b2 * 10

    def test_recency_decays(self):
        now = int(time.time())
        recent = reinforcement_boost({
            "mention_count": 5, "last_reinforced_at": now - 60,  # 1 minute ago
        })
        old = reinforcement_boost({
            "mention_count": 5, "last_reinforced_at": now - 365 * 86400,  # 1 year ago
        })
        assert recent > old

    def test_malformed_metadata_falls_back(self):
        # mention_count is a string → default 1
        b = reinforcement_boost({"mention_count": "bogus"})
        assert b == 0.0

    # ----- Phase A-2: temporal age decay -----

    def test_age_decay_recent_fact_has_higher_boost(self):
        now = int(time.time())
        recent = reinforcement_boost({"timestamp": now - 60})
        old = reinforcement_boost({"timestamp": now - 365 * 86400})
        # Both should be > 0 but recent should outweigh old.
        assert recent > old > 0.0

    def test_age_decay_zero_when_no_timestamp(self):
        # No timestamp / created_at_unix → age component disabled.
        b = reinforcement_boost({"mention_count": 1})
        assert b == 0.0

    def test_age_decay_uses_created_at_unix_when_present(self):
        now = int(time.time())
        # created_at_unix takes precedence over timestamp.
        b1 = reinforcement_boost({"created_at_unix": now - 1})
        b2 = reinforcement_boost({"timestamp": now - 1})
        assert pytest.approx(b1, abs=1e-6) == b2

    def test_age_decay_can_be_disabled_via_argument(self):
        now = int(time.time())
        # alpha_age=0 disables the age component, leaving only mention/recency.
        b_off = reinforcement_boost(
            {"timestamp": now - 60, "mention_count": 1}, alpha_age=0.0,
        )
        assert b_off == 0.0

    def test_combined_boost_stays_within_reasonable_range(self):
        # Heavy fact: 100 mentions, just reinforced, brand new.
        now = int(time.time())
        b = reinforcement_boost({
            "mention_count": 100,
            "last_reinforced_at": now,
            "timestamp": now,
        })
        # Loose upper bound — three components, each capped by α coefficient.
        # mention ≈ 0.10 * log(100) ≈ 0.46, recency ≈ 0.05, age ≈ 0.03.
        # Asserting < 1.0 prevents future regressions where someone bumps α
        # coefficients without thinking about totals.
        assert 0.3 < b < 1.0


class TestReinforceOnRecallSync:
    """Phase ①: testing-effect reinforcement on search-time surfacing."""

    def test_bumps_recall_count_not_mention_count(self):
        mem = MagicMock()
        mem.get.return_value = {
            "id": "m1", "memory": "Rust が好き",
            "metadata": {"mention_count": 1, "recall_count": 0},
        }
        ok = reinforce_on_recall_sync(mem, "m1")
        assert ok is True
        partial = mem.update.call_args.kwargs["metadata"]
        assert partial["recall_count"] == 1
        assert "mention_count" not in partial  # separate exposure channel
        assert "importance" not in partial

    def test_rate_limited_within_spacing_window(self):
        now = int(time.time())
        mem = MagicMock()
        mem.get.return_value = {
            "id": "m1", "memory": "x",
            "metadata": {"recall_count": 1, "last_recalled_at": now},
        }
        ok = reinforce_on_recall_sync(mem, "m1")
        assert ok is False
        mem.update.assert_not_called()

    def test_allowed_after_spacing_window(self):
        stale = int(time.time()) - 7 * 3600  # older than the 6h default window
        mem = MagicMock()
        mem.get.return_value = {
            "id": "m1", "memory": "x",
            "metadata": {"recall_count": 1, "last_recalled_at": stale},
        }
        ok = reinforce_on_recall_sync(mem, "m1")
        assert ok is True
        assert mem.update.call_args.kwargs["metadata"]["recall_count"] == 2

    def test_spacing_window_is_dynamic_setting(self, monkeypatch):
        # settings are read per call — a shortened window takes effect
        # without any reload.
        monkeypatch.setenv("MEM0_RECALL_REINFORCE_MIN_INTERVAL_SEC", "10")
        stale = int(time.time()) - 60
        mem = MagicMock()
        mem.get.return_value = {
            "id": "m1", "memory": "x",
            "metadata": {"recall_count": 1, "last_recalled_at": stale},
        }
        assert reinforce_on_recall_sync(mem, "m1") is True


class TestActrActivation:
    """Phase ②: ACT-R base-level activation boost."""

    def test_no_presentations_returns_neg_inf(self):
        result = _actr_activation(0, [], mention_count=0, recall_count=0, now=time.time())
        assert result == float("-inf")

    def test_single_recent_presentation_beats_single_old_one(self):
        now = time.time()
        recent = _actr_activation(int(now - 3600), [], mention_count=1, recall_count=0, now=now)
        old = _actr_activation(int(now - 86400 * 365), [], mention_count=1, recall_count=0, now=now)
        assert recent > old

    def test_more_presentations_increase_activation(self):
        now = time.time()
        created = int(now - 86400 * 10)
        few = _actr_activation(created, [int(now - 3600)], mention_count=2, recall_count=0, now=now)
        many = _actr_activation(
            created, [int(now - 3600 * h) for h in (1, 2, 3, 4, 5)],
            mention_count=6, recall_count=0, now=now,
        )
        assert many > few

    def test_evicted_history_approximation_is_finite_and_monotonic(self):
        now = time.time()
        created = int(now - 86400 * 60)
        known = [int(now - 86400 * d) for d in (50, 40, 30, 20, 10, 5, 4, 3, 2, 1)]
        low_n = _actr_activation(created, known, mention_count=10, recall_count=0, now=now)
        high_n = _actr_activation(created, known, mention_count=40, recall_count=0, now=now)
        assert math.isfinite(low_n)
        assert math.isfinite(high_n)
        assert high_n >= low_n

    def test_boost_dispatch_default_is_legacy(self, monkeypatch):
        monkeypatch.delenv("MEM0_BOOST_MODE", raising=False)
        val = reinforcement_boost({"mention_count": 5, "last_reinforced_at": int(time.time())})
        assert val > 0.0

    def test_boost_dispatch_actr_mode(self, monkeypatch):
        monkeypatch.setenv("MEM0_BOOST_MODE", "actr")
        now = time.time()
        md = {
            "timestamp": int(now - 86400 * 5),
            "mention_count": 3, "recall_count": 1,
            "reinforced_at": [int(now - 3600)],
        }
        val = reinforcement_boost(md)
        assert math.isclose(val, _actr_reinforcement_boost(md), rel_tol=1e-6)

    def test_actr_boost_in_valid_range(self):
        now = time.time()
        md = {
            "timestamp": int(now - 86400 * 5),
            "mention_count": 3, "recall_count": 2,
            "reinforced_at": [int(now - 3600), int(now - 7200)],
        }
        boost = _actr_reinforcement_boost(md)
        assert 0.0 < boost < 0.2 + 1e-9
