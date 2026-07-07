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
import tempfile
import threading
import time

import aiohttp

from .base import AuthBackend

log = logging.getLogger(__name__)

CREDENTIALS_PATH = os.environ.get(
    "CLAUDE_CLI_CREDENTIALS_PATH", os.path.expanduser("~/.claude/.credentials.json"),
)
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

# Any process that shares the ~/.claude directory with us could otherwise
# race our atomic-rename (TOCTOU) or read a refreshed token off disk. Refuse
# to write credentials back unless the directory AND the existing file are
# already restricted to the current user.
_CREDS_DIR_MODE_MASK = 0o077  # group/other must have no bits set
_CREDS_FILE_MODE_MASK = 0o077


def _verify_path_privacy(path: str, *, is_dir: bool) -> bool:
    """Return True iff `path` is owned by us and g/o perms are clear."""
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
    mode = stat.S_IMODE(st.st_mode)
    mask = _CREDS_DIR_MODE_MASK if is_dir else _CREDS_FILE_MODE_MASK
    if mode & mask:
        log.error(
            "Refusing to touch %s: mode=%#o grants g/o access (mask %#o)",
            path, mode, mask,
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


# macOS: the Claude Code CLI stores its OAuth credentials in the login
# Keychain (as a generic password under this service name) rather than in the
# plaintext JSON file, so on darwin we read the Keychain first and fall back to
# the file. Set LLM_MEM0_DISABLE_KEYCHAIN=1 to force the file path.
_KEYCHAIN_SERVICE = os.environ.get("CLAUDE_CLI_KEYCHAIN_SERVICE", "Claude Code-credentials")


def _read_credentials_from_keychain() -> dict | None:
    """On macOS, read the Claude Code OAuth blob from the login Keychain.

    Returns the ``claudeAiOauth`` sub-dict, or None when the entry is absent,
    access is denied, or we're not on macOS. Best-effort — never raises.
    """
    if sys.platform != "darwin" or os.environ.get("LLM_MEM0_DISABLE_KEYCHAIN") == "1":
        return None
    import subprocess
    try:
        # -w prints only the secret value (the JSON blob) to stdout.
        out = subprocess.run(
            ["security", "find-generic-password", "-s", _KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        data = json.loads(out.stdout)
        return data.get("claudeAiOauth")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        log.debug("Keychain credential read failed (non-fatal): %s", exc)
        return None


def _read_credentials(path: str | None = None) -> dict | None:
    if path is None:
        path = CREDENTIALS_PATH
    kc = _read_credentials_from_keychain()
    if kc and kc.get("accessToken"):
        return kc
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("claudeAiOauth")
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        log.warning("Failed to read Claude Code CLI credentials from %s: %s", path, exc)
        return None


def _save_credentials(access: str, refresh: str, expires_at: int, *, path: str | None = None) -> None:
    """Write refreshed tokens back to the credentials file (atomic write).

    Defensive ordering: verify the directory and file are privately-owned
    before touching them, then fchmod the new fd to 0o600 BEFORE any content
    lands in it, so there's no window where a co-tenant could read a
    plaintext token off an unprotected tempfile.
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

        fd, tmp_path = tempfile.mkstemp(dir=creds_dir, suffix=".tmp")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
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
        self._http_session: aiohttp.ClientSession | None = None
        self._client = None
        self._client_token: str | None = None
        self._client_lock = asyncio.Lock()

        if not self._load_from_disk():
            raise RuntimeError(f"No Claude Code CLI credentials found at {CREDENTIALS_PATH}")

        _active_token_getter = lambda: self._access_token  # noqa: E731
        _patch_anthropic_sdk()

    def _load_from_disk(self) -> bool:
        creds = _read_credentials(CREDENTIALS_PATH)
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
        # Still expired: refresh ourselves as a last resort. This ROTATES the
        # refresh token, which invalidates the copy the Claude Code CLI holds —
        # the user may then need to `claude` re-login. We only reach here when
        # the CLI has not refreshed the token for us.
        if self._is_expired() and self._refresh_token:
            log.warning(
                "Claude Code CLI token expired and the CLI has not refreshed it; "
                "refreshing directly (this may require a CLI re-login afterward)."
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
        return os.environ.get("MEM0_LLM_MODEL", "claude-haiku-4-5-20251001")

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
