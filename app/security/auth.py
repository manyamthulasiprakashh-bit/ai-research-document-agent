"""Trusted user authentication for the local research API.

Identity is resolved on the server from a shared secret. A client only supplies
a secret API key; it can never choose its own user ID. The server maps each
configured key to an owner identity using the ``API_KEYS`` environment variable.
"""

import hmac
import logging
import os
import threading
import time
from dataclasses import dataclass

from fastapi import Header, HTTPException, Request, status

logger = logging.getLogger(__name__)

API_KEYS_ENV = "API_KEYS"
DEFAULT_DENIAL_AUDIT_INTERVAL_SECONDS = 5.0


@dataclass(frozen=True)
class AuthenticatedUser:
    """A server-resolved identity for one authenticated request."""

    user_id: str


def load_api_keys(raw: str | None = None) -> dict[str, str]:
    """Parse ``API_KEYS`` into a mapping of secret key -> owner user ID.

    The expected format is ``user-a:key-a,user-b:key-b``. Entries without a
    colon or with empty parts are ignored so a malformed value cannot create a
    wildcard identity. A secret that maps to two different owners is ambiguous,
    so it is dropped entirely and can never authenticate.
    """
    configured = raw if raw is not None else os.getenv(API_KEYS_ENV, "")
    keys: dict[str, str] = {}
    ambiguous: set[str] = set()
    malformed = 0

    for entry in configured.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            malformed += 1
            continue
        user_id, secret = entry.split(":", 1)
        user_id = user_id.strip()
        secret = secret.strip()
        if not user_id or not secret:
            malformed += 1
            continue
        if secret in ambiguous:
            continue
        existing_owner = keys.get(secret)
        if existing_owner is not None and existing_owner != user_id:
            ambiguous.add(secret)
            keys.pop(secret, None)
            continue
        keys[secret] = user_id

    if malformed:
        logger.warning(
            "Ignored %d malformed %s entr%s (missing ':' or empty owner/secret).",
            malformed,
            API_KEYS_ENV,
            "y" if malformed == 1 else "ies",
        )
    if ambiguous:
        logger.warning(
            "Dropped %d ambiguous %s secret%s that mapped to multiple owners.",
            len(ambiguous),
            API_KEYS_ENV,
            "" if len(ambiguous) == 1 else "s",
        )

    return keys


def resolve_user(api_key: str | None, keys: dict[str, str]) -> AuthenticatedUser | None:
    """Return the owner identity for a secret key, comparing in constant time."""
    if not api_key:
        return None

    for secret, user_id in keys.items():
        if hmac.compare_digest(secret, api_key):
            return AuthenticatedUser(user_id)
    return None


class AccessDeniedThrottle:
    """Limit how often denied-access audit writes happen in this process.

    This bounds audit-table growth caused by repeated unauthenticated requests.
    It does NOT rate-limit requests themselves (every request still gets a
    ``401``), and it is per-process: each worker keeps its own state.
    """

    def __init__(
        self,
        interval_seconds: float = DEFAULT_DENIAL_AUDIT_INTERVAL_SECONDS,
        clock=time.monotonic,
    ) -> None:
        self._interval_seconds = interval_seconds
        self._clock = clock
        self._last_allowed: float | None = None
        self._lock = threading.Lock()

    def allow(self, key: str = "") -> bool:
        """Return ``True`` at most once per interval, ignoring ``key``.

        ``key`` is accepted so callers can pass the request path; the limit is a
        single process-wide bucket, so the value does not change the decision.
        """
        now = self._clock()
        with self._lock:
            if (
                self._last_allowed is not None
                and now - self._last_allowed < self._interval_seconds
            ):
                return False
            self._last_allowed = now
            return True


_DENIAL_THROTTLE = AccessDeniedThrottle()


def _record_access_denied(request: Request) -> None:
    """Best-effort audit of a denied request without leaking the supplied key."""
    audit_log = getattr(request.app.state, "audit_log", None)
    if audit_log is None:
        return
    throttle = getattr(request.app.state, "denial_throttle", _DENIAL_THROTTLE)
    if not throttle.allow(request.url.path):
        return
    try:
        audit_log.record(
            actor="anonymous",
            action="access_denied",
            resource_id=request.url.path,
            outcome="denied",
            detail="invalid_or_missing_credentials",
        )
    except Exception:
        pass


async def require_user(
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> AuthenticatedUser:
    """FastAPI dependency enforcing a valid server-side API key."""
    keys: dict[str, str] = getattr(request.app.state, "api_keys", {})
    user = resolve_user(x_api_key, keys)
    if user is None:
        _record_access_denied(request)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return user
