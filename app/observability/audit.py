"""Append-only audit trail for security-relevant document and session actions.

Events record an identifier, actor, action, resource, outcome, and timestamp.
Passwords, API keys, tokens, and full document contents are never stored here.
"""

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


class AuditLog:
    """Persist small audit events in SQLite for later inspection."""

    def __init__(self, path: str | Path = "data/storage/audit.db") -> None:
        """Open or create the audit database."""
        self._database_path = Path(path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self._database_path,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_actor
                    ON audit_events(actor)
                """
            )

    def record(
        self,
        actor: str,
        action: str,
        resource_id: str = "",
        outcome: str = "success",
        detail: str = "",
    ) -> dict[str, str]:
        """Append one audit event and return it."""
        event = {
            "event_id": str(uuid.uuid4()),
            "actor": str(actor),
            "action": str(action),
            "resource_id": str(resource_id),
            "outcome": str(outcome),
            "detail": str(detail),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO audit_events
                    (event_id, actor, action, resource_id, outcome, detail,
                     created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["event_id"],
                    event["actor"],
                    event["action"],
                    event["resource_id"],
                    event["outcome"],
                    event["detail"],
                    event["created_at"],
                ),
            )
        return event

    def list_events(
        self,
        actor: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, str]]:
        """Return recent events, optionally limited to one actor."""
        with self._lock:
            if actor is None:
                rows = self._connection.execute(
                    """
                    SELECT event_id, actor, action, resource_id, outcome,
                           detail, created_at
                    FROM audit_events
                    ORDER BY rowid DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
            else:
                rows = self._connection.execute(
                    """
                    SELECT event_id, actor, action, resource_id, outcome,
                           detail, created_at
                    FROM audit_events
                    WHERE actor = ?
                    ORDER BY rowid DESC
                    LIMIT ?
                    """,
                    (actor, int(limit)),
                ).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        """Return the number of recorded audit events."""
        with self._lock:
            return self._connection.execute(
                "SELECT COUNT(*) FROM audit_events"
            ).fetchone()[0]

    def close(self) -> None:
        """Close the audit database connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "AuditLog":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
