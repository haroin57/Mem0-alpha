"""Tests for the auth-backend injection API (set_auth_backend) and the
client hooks it powers: provider-drift rebuild and refresh_memory_llm.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_mem0 import auth, client
from llm_mem0.auth.base import AuthBackend


class FakeBackend(AuthBackend):
    def __init__(self, provider: str = "fake"):
        self._provider = provider
        self.refreshed_with: list = []

    async def complete(self, *, system, user_message, model, max_tokens) -> str:
        return ""

    def mem0_llm_config(self, *, model) -> dict:
        return {"provider": self._provider, "config": {"model": model}}

    def default_model(self) -> str:
        return "fake-model"

    def provider_id(self) -> str:
        return self._provider

    def refresh_memory_llm(self, memory) -> None:
        self.refreshed_with.append(memory)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    auth.reset_auth_backend()
    monkeypatch.setattr(client, "_mem0_instance", None)
    monkeypatch.setattr(client, "_mem0_provider_id", None)
    monkeypatch.setattr(client, "_init_failed_at", None)
    yield
    auth.reset_auth_backend()


class TestSetAuthBackend:
    def test_injection_wins_over_autodetect(self):
        backend = FakeBackend()
        auth.set_auth_backend(backend)
        assert auth.get_auth_backend() is backend

    def test_reset_clears_injection(self):
        auth.set_auth_backend(FakeBackend())
        auth.reset_auth_backend()
        assert auth._backend is None

    def test_default_provider_id_is_class_name(self):
        class Minimal(AuthBackend):
            async def complete(self, **kw):
                return ""

            def mem0_llm_config(self, *, model):
                return {}

            def default_model(self):
                return "m"

        assert Minimal().provider_id() == "Minimal"


class TestClientHooks:
    def test_refresh_memory_llm_called_on_cached_handout(self, monkeypatch):
        backend = FakeBackend()
        auth.set_auth_backend(backend)
        fake_mem = MagicMock()
        monkeypatch.setattr(client, "_mem0_instance", fake_mem)
        monkeypatch.setattr(client, "_mem0_provider_id", "fake")

        out = client._get_mem0()

        assert out is fake_mem
        assert backend.refreshed_with == [fake_mem]

    def test_provider_drift_rebuilds_instance(self, monkeypatch):
        backend = FakeBackend(provider="openrouter")
        auth.set_auth_backend(backend)
        stale = MagicMock(name="stale")
        rebuilt = MagicMock(name="rebuilt")
        monkeypatch.setattr(client, "_mem0_instance", stale)
        # instance was built while the provider was "anthropic" — drift.
        monkeypatch.setattr(client, "_mem0_provider_id", "anthropic")
        monkeypatch.setattr(client, "_ping_chromadb_if_server_mode", lambda: None)
        monkeypatch.setattr(client, "_build_config", lambda: {
            "llm": {"provider": "openrouter", "config": {"model": "m"}},
            "vector_store": {"config": {"host": "h", "port": 1}},
            "embedder": {"config": {"model": "e"}},
        })
        import mem0
        monkeypatch.setattr(
            mem0.Memory, "from_config", classmethod(lambda cls, cfg: rebuilt))

        out = client._get_mem0()

        assert out is rebuilt
        assert client._mem0_provider_id == "openrouter"
        # the rebuilt instance got the refresh hook too
        assert rebuilt in backend.refreshed_with

    def test_no_drift_no_rebuild(self, monkeypatch):
        backend = FakeBackend(provider="fake")
        auth.set_auth_backend(backend)
        fake_mem = MagicMock()
        monkeypatch.setattr(client, "_mem0_instance", fake_mem)
        monkeypatch.setattr(client, "_mem0_provider_id", "fake")

        rebuild_spy = MagicMock(side_effect=AssertionError("must not rebuild"))
        monkeypatch.setattr(client, "_build_config", rebuild_spy)

        assert client._get_mem0() is fake_mem
        rebuild_spy.assert_not_called()

    def test_erroring_hooks_never_break_handout(self, monkeypatch):
        class Broken(FakeBackend):
            def provider_id(self):
                raise RuntimeError("boom")

            def refresh_memory_llm(self, memory):
                raise RuntimeError("boom")

        auth.set_auth_backend(Broken())
        fake_mem = MagicMock()
        monkeypatch.setattr(client, "_mem0_instance", fake_mem)
        monkeypatch.setattr(client, "_mem0_provider_id", "fake")

        assert client._get_mem0() is fake_mem
