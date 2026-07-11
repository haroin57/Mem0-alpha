"""Reuse an existing Claude Code CLI OAuth session for Anthropic calls.

Reads tokens from ``~/.claude/.credentials.json`` — the same file the
``claude`` CLI itself reads and refreshes — so a developer who is already
logged into Claude Code doesn't need to provision a separate Anthropic API
key just to use this library.

This reuses your own already-authenticated CLI session; it does not attempt
to impersonate the CLI's network fingerprint. Requests are made with the
standard Anthropic Python SDK — no header/TLS spoofing is performed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import stat
import sys
import threading
import time

import aiohttp

from ..atomic_write import atomic_write_json
from ..settings import DEFAULT_HELPER_MODEL, settings
from .base import AuthBackend

log = logging.getLogger(__name__)

def _resolve_credentials_path() -> str:
    """Where the Claude Code CLI keeps its login credentials.

    Same JSON file on Linux, macOS, and Windows: under ``$CLAUDE_CONFIG_DIR``
    when set, otherwise ``~/.claude``. ``CLAUDE_CLI_CREDENTIALS_PATH`` overrides
    everything. (On macOS the CLI *also* mirrors these into the Keychain, which
    we read first — see below.)
    """
    override = os.environ.get("CLAUDE_CLI_CREDENTIALS_PATH")
    if override:
        return override
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(
        os.path.expanduser("~"), ".claude"
    )
    return os.path.join(config_dir, ".credentials.json")


CREDENTIALS_PATH = _resolve_credentials_path()
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
OAUTH_BETA_HEADER = "oauth-2025-04-20"
REFRESH_SCOPES = [
    "user:profile",
    "user:inference",
    "user:sessions:claude_code",
    "user:mcp_servers",
    "user:file_upload",
]
# Refresh 5 minutes before expiry.
EXPIRY_BUFFER_MS = 5 * 60 * 1000

# We write a refreshed token back only into a path we own. The *file* must
# additionally not be group/other-readable (it holds the token). The directory
# is only required to be owned by us — ``~/.claude`` is created 0o755 by the
# CLI on a normal setup, and demanding 0o700 there would make token persistence
# fail on virtually every machine (which in turn orphans the CLI's own token).
_CREDS_FILE_MODE_MASK = 0o077  # group/other must have no bits set on the file


def _verify_path_privacy(path: str, *, is_dir: bool) -> bool:
    """Return True iff we own `path` (and, for a file, g/o perms are clear).

    POSIX only. Windows has no uid/mode bits to check — the credentials file
    inherits the ACLs of the user profile directory — so the check is skipped
    there and we rely on the atomic ``os.replace`` write below.
    """
    if os.name != "posix":
        return True
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return True  # nothing to verify yet — the first write will create it
    except OSError as exc:
        log.error("Cannot stat %s: %s", path, exc)
        return False
    if st.st_uid != os.getuid():
        log.error(
            "Refusing to touch %s: owner=%d but we are uid=%d",
            path, st.st_uid, os.getuid(),
        )
        return False
    if not is_dir:
        mode = stat.S_IMODE(st.st_mode)
        if mode & _CREDS_FILE_MODE_MASK:
            log.error(
                "Refusing to touch %s: mode=%#o grants g/o access (mask %#o)",
                path, mode, _CREDS_FILE_MODE_MASK,
            )
            return False
    return True


def _oauth_headers(extra: dict | None = None) -> dict:
    """Headers required for OAuth Bearer auth against the Anthropic API.

    ``anthropic-beta: oauth-2025-04-20`` is mandatory — an OAuth token without
    it is rejected with HTTP 401. ``anthropic-dangerous-direct-browser-access``
    matches what the CLI itself sends on OAuth requests; its absence has been
    observed to trigger HTTP 429 on the first call.
    """
    out = dict(extra or {})
    existing_beta = out.get("anthropic-beta", "")
    if OAUTH_BETA_HEADER not in existing_beta:
        out["anthropic-beta"] = (
            f"{existing_beta},{OAUTH_BETA_HEADER}".strip(",")
            if existing_beta else OAUTH_BETA_HEADER
        )
    out.setdefault("anthropic-dangerous-direct-browser-access", "true")
    return out


# NOTE on the Keychain flags (settings.CLAUDE_CLI_KEYCHAIN_SERVICE,
# settings.LLM_MEM0_DISABLE_KEYCHAIN, settings.LLM_MEM0_ALLOW_TOKEN_REFRESH):
# macOS keeps the CLI's OAuth credentials in the login Keychain rather than
# the plaintext JSON file, so on darwin we read the Keychain first and fall
# back to the file. A direct token refresh ROTATES the refresh token, which
# invalidates the copy the Claude Code CLI holds unless we can write the new
# one back to the same store. We can do that for the file store; we cannot
# reliably do it for the macOS Keychain. So when the token came from the
# Keychain we do NOT refresh by default — we rely on the CLI's own refresh
# cycle (re-reading the store) and only refresh directly if the user opts in
# via LLM_MEM0_ALLOW_TOKEN_REFRESH=1, accepting a possible CLI re-login.


def _read_credentials_from_keychain() -> dict | None:
    """On macOS, read the Claude Code OAuth blob from the login Keychain.

    Returns the ``claudeAiOauth`` sub-dict, or None when the entry is absent,
    access is denied, or we're not on macOS. Best-effort — never raises.
    """
    if sys.platform != "darwin" or settings.LLM_MEM0_DISABLE_KEYCHAIN:
        return None
    import subprocess
    try:
        # -w prints only the secret value (the JSON blob) to stdout.
        out = subprocess.run(
            ["security", "find-generic-password",
             "-s", settings.CLAUDE_CLI_KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        data = json.loads(out.stdout)
        return data.get("claudeAiOauth")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        log.debug("Keychain credential read failed (non-fatal): %s", exc)
        return None


def _read_credentials_from_file(path: str | None = None) -> dict | None:
    if path is None:
        path = CREDENTIALS_PATH
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("claudeAiOauth")
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        log.warning("Failed to read Claude Code CLI credentials from %s: %s", path, exc)
        return None


def _save_credentials(access: str, refresh: str, expires_at: int, *, path: str | None = None) -> None:
    """Write refreshed tokens back to the credentials file (atomic write).

    On POSIX, defensive ordering: verify the directory and file are
    privately-owned before touching them, then hand off to
    ``atomic_write_json`` (0o600 before content, fsync, ``os.replace``) so
    there's no window where a co-tenant could read a plaintext token off an
    unprotected tempfile. On Windows the mode is a no-op — the file inherits
    the user profile's ACLs — and the atomic replace still applies.
    """
    if path is None:
        path = CREDENTIALS_PATH
    try:
        creds_dir = os.path.dirname(path)
        if not _verify_path_privacy(creds_dir, is_dir=True):
            log.error("Refusing to save credentials: %s is not private", creds_dir)
            return
        if not _verify_path_privacy(path, is_dir=False):
            log.error("Refusing to overwrite credentials: %s is not private", path)
            return

        try:
            with open(path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}

        oauth = data.get("claudeAiOauth", {})
        oauth["accessToken"] = access
        oauth["refreshToken"] = refresh
        oauth["expiresAt"] = expires_at
        data["claudeAiOauth"] = oauth

        atomic_write_json(path, data, mode=0o600, indent=None)
    except Exception as exc:
        log.warning("Failed to save Claude Code CLI credentials: %s", exc)


# ---------------------------------------------------------------------------
# SDK patching — shared process-wide state (patching is a global SDK effect)
# ---------------------------------------------------------------------------
_patched = False
_orig_anthropic_init = None
_patch_lock = threading.Lock()
_active_token_getter = None  # set by the most recently constructed instance


def _patch_anthropic_sdk() -> None:
    """Monkey-patch anthropic.Anthropic so instances use OAuth Bearer auth.

    mem0 internally does ``anthropic.Anthropic(api_key=...)``, which sets the
    x-api-key header. We intercept this to drop api_key and inject
    auth_token + the required beta header instead. Thread-safe
    double-checked locking: two concurrent callers can't both race past the
    unlocked check and patch twice.
    """
    global _patched
    if _patched:
        return
    with _patch_lock:
        if _patched:
            return
        _patch_anthropic_sdk_locked()


def _patch_anthropic_sdk_locked() -> None:
    global _patched, _orig_anthropic_init
    try:
        import anthropic

        _orig_anthropic_init = anthropic.Anthropic.__init__

        def _oauth_init(self, **kwargs):
            kwargs.pop("api_key", None)
            kwargs["auth_token"] = _active_token_getter() if _active_token_getter else None
            headers = dict(kwargs.pop("default_headers", None) or {})
            existing_beta = headers.get("anthropic-beta", "")
            if OAUTH_BETA_HEADER not in existing_beta:
                beta = (
                    f"{existing_beta},{OAUTH_BETA_HEADER}".strip(",")
                    if existing_beta else OAUTH_BETA_HEADER
                )
                headers["anthropic-beta"] = beta
            kwargs["default_headers"] = headers
            _orig_anthropic_init(self, **kwargs)

        anthropic.Anthropic.__init__ = _oauth_init
        _patched = True
        log.info("Anthropic SDK patched for OAuth Bearer auth (Claude Code CLI session)")
    except ImportError:
        log.warning("anthropic package not installed, skipping SDK patch")


class AnthropicCliAuth(AuthBackend):
    """Reuses ``~/.claude/.credentials.json`` (the Claude Code CLI's own OAuth
    session) for both mem0's internal Anthropic calls and this library's own
    direct helper calls (classification/extraction).
    """

    def __init__(self) -> None:
        global _active_token_getter
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._expires_at: int = 0
        self._source: str = "file"  # "file" | "keychain" — where we read the token
        self._http_session: aiohttp.ClientSession | None = None
        self._client = None
        self._client_token: str | None = None
        self._client_lock = asyncio.Lock()

        if not self._load_from_disk():
            raise RuntimeError(f"No Claude Code CLI credentials found at {CREDENTIALS_PATH}")

        _active_token_getter = lambda: self._access_token  # noqa: E731
        _patch_anthropic_sdk()

    def _load_from_disk(self) -> bool:
        # Prefer the Keychain (macOS) and record which store the token came
        # from — it decides whether we may safely refresh (a rotating refresh
        # is only safe when we can write the new token back to the same store).
        kc = _read_credentials_from_keychain()
        if kc and kc.get("accessToken"):
            creds, self._source = kc, "keychain"
        else:
            creds, self._source = _read_credentials_from_file(CREDENTIALS_PATH), "file"
        if not creds or not creds.get("accessToken"):
            return False
        self._access_token = creds.get("accessToken")
        self._refresh_token = creds.get("refreshToken")
        self._expires_at = creds.get("expiresAt", 0)
        return True

    def _is_expired(self) -> bool:
        now_ms = int(time.time() * 1000)
        return (now_ms + EXPIRY_BUFFER_MS) >= self._expires_at

    async def _get_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def _refresh(self) -> bool:
        if not self._refresh_token:
            return False
        body = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": CLIENT_ID,
            "scope": " ".join(REFRESH_SCOPES),
        }
        try:
            session = await self._get_http_session()
            async with session.post(
                TOKEN_URL, json=body,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    log.warning("Claude Code CLI token refresh failed: HTTP %d: %s", resp.status, text)
                    return False
                data = await resp.json()
            self._access_token = data["access_token"]
            self._refresh_token = data.get("refresh_token", self._refresh_token)
            self._expires_at = int(time.time() * 1000) + data["expires_in"] * 1000
            _save_credentials(self._access_token, self._refresh_token, self._expires_at)
            log.info("Claude Code CLI OAuth token refreshed, expires in %ds", data["expires_in"])
            return True
        except Exception as exc:
            log.warning("Claude Code CLI token refresh failed: %s", exc)
            return False

    async def _ensure_valid_token(self) -> str | None:
        if self._access_token and not self._is_expired():
            return self._access_token
        # Expired: first re-read the source (Keychain/file). If the Claude Code
        # CLI is running it keeps its own token fresh, so this picks up a valid
        # token without us rotating anything.
        if self._load_from_disk() and not self._is_expired():
            return self._access_token
        # Still expired. Refreshing ourselves rotates the refresh token; that's
        # only safe when we can persist the new one back to the same store the
        # CLI reads. We can for the file store, but not reliably for the macOS
        # Keychain — so a Keychain-sourced token is refreshed only on explicit
        # opt-in. Otherwise we leave the CLI's token untouched and return what
        # we have (callers treat an unusable token as "no memory this turn").
        if self._is_expired() and self._refresh_token:
            if self._source == "keychain" and not settings.LLM_MEM0_ALLOW_TOKEN_REFRESH:
                log.warning(
                    "Claude Code CLI token (from Keychain) is expired and the CLI "
                    "hasn't refreshed it. Not refreshing directly, to avoid "
                    "invalidating the CLI's login. Run any `claude` command to "
                    "refresh it, or set LLM_MEM0_ALLOW_TOKEN_REFRESH=1 to override."
                )
            else:
                log.warning(
                    "Claude Code CLI token expired and the CLI has not refreshed "
                    "it; refreshing directly%s.",
                    " (Keychain source — this may require a CLI re-login afterward)"
                    if self._source == "keychain" else "",
                )
                if await self._refresh():
                    return self._access_token
        return self._access_token

    async def _get_client(self):
        token = await self._ensure_valid_token()
        if token is None:
            return None
        if self._client is not None and token == self._client_token:
            return self._client
        async with self._client_lock:
            if self._client is None or token != self._client_token:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(
                    auth_token=token, max_retries=0,
                    default_headers=_oauth_headers(),
                )
                self._client_token = token
        return self._client

    def default_model(self) -> str:
        return settings.MEM0_LLM_MODEL or DEFAULT_HELPER_MODEL

    def mem0_llm_config(self, *, model: str) -> dict:
        # No api_key here — the SDK patch injects auth_token when mem0
        # constructs its own anthropic.Anthropic(...) internally.
        return {
            "provider": "anthropic",
            "config": {"model": model, "temperature": 0.1, "max_tokens": 2000},
        }

    async def complete(self, *, system: str, user_message: str, model: str, max_tokens: int) -> str:
        try:
            client = await self._get_client()
            if client is None:
                return ""
            resp = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_message}],
            )
            for block in getattr(resp, "content", []) or []:
                text = getattr(block, "text", None)
                if text:
                    return text
            return ""
        except Exception as exc:
            log.warning("AnthropicCliAuth.complete failed: %s", exc)
            return ""

    async def aclose(self) -> None:
        """Close the cached aiohttp session and Anthropic client.

        The backend is a process-wide singleton, so this matters mainly for
        embedded hosts and tests that build/tear down event loops — without
        it every loop teardown logs "Unclosed client session".
        """
        session, self._http_session = self._http_session, None
        if session is not None and not session.closed:
            try:
                await session.close()
            except Exception as exc:
                log.debug("AnthropicCliAuth.aclose(session): %s", exc)
        client, self._client = self._client, None
        self._client_token = None
        if client is not None:
            try:
                await client.close()
            except Exception as exc:
                log.debug("AnthropicCliAuth.aclose(client): %s", exc)
