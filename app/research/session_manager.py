import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

PUBLIC_SCOPE = "public"
LEGACY_SCOPE = "legacy"


class SessionNotFoundError(LookupError):
    """Raised when a research session does not exist or is not owned."""


class ResearchSessionManager:
    """Persist owner-scoped research sessions and their results in SQLite."""

    def __init__(self, path: str | Path = "data/storage/research_sessions.db") -> None:
        """Open or create the research session database."""
        self._database_path = Path(path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self._database_path,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        self._migrate()

    @staticmethod
    def _scope(owner_id: str | None) -> str:
        """Return the normalized owner scope for an optional owner id."""
        return owner_id or PUBLIC_SCOPE

    def _create_tables(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS research_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id)
                        ON DELETE CASCADE,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT NOT NULL,
                    used_local_documents INTEGER NOT NULL,
                    used_web_search INTEGER NOT NULL,
                    errors TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_results_session_id
                    ON research_results(session_id);
                """
            )

    def _migrate(self) -> None:
        """Add ownership to session databases created before this feature."""
        with self._lock, self._connection:
            columns = {
                row["name"]
                for row in self._connection.execute(
                    "PRAGMA table_info(sessions)"
                ).fetchall()
            }
            if "owner_id" in columns:
                return
            self._connection.execute(
                "ALTER TABLE sessions ADD COLUMN owner_id TEXT NOT NULL "
                f"DEFAULT '{LEGACY_SCOPE}'"
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_session(self, title: str, owner_id: str | None = None) -> dict[str, str]:
        """Create and persist a new research session for one owner."""
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("title cannot be empty")

        session = {
            "session_id": str(uuid.uuid4()),
            "title": clean_title,
            "created_at": self._now(),
        }
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO sessions
                    (session_id, title, owner_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session["session_id"],
                    session["title"],
                    self._scope(owner_id),
                    session["created_at"],
                    session["created_at"],
                ),
            )
        return session

    def list_sessions(self, owner_id: str | None = None) -> list[dict[str, str]]:
        """Return sessions owned by one owner, most recently updated first."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT session_id, title, created_at, updated_at
                FROM sessions
                WHERE owner_id = ?
                ORDER BY updated_at DESC
                """,
                (self._scope(owner_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_session(
        self,
        session_id: str,
        owner_id: str | None = None,
    ) -> dict[str, object]:
        """Return one owned session's metadata and its research history."""
        scope = self._scope(owner_id)
        with self._lock:
            session_row = self._connection.execute(
                """
                SELECT session_id, title, created_at, updated_at
                FROM sessions
                WHERE session_id = ? AND owner_id = ?
                """,
                (session_id, scope),
            ).fetchone()
            if not session_row:
                raise SessionNotFoundError(f"Session not found: {session_id}")

            result_rows = self._connection.execute(
                """
                SELECT question, answer, sources, used_local_documents,
                       used_web_search, errors, created_at
                FROM research_results
                WHERE session_id = ?
                ORDER BY result_id ASC
                """,
                (session_id,),
            ).fetchall()

        session = dict(session_row)
        session["questions"] = [
            {
                "question": row["question"],
                "answer": row["answer"],
                "sources": json.loads(row["sources"]),
                "used_local_documents": bool(row["used_local_documents"]),
                "used_web_search": bool(row["used_web_search"]),
                "errors": json.loads(row["errors"]),
                "created_at": row["created_at"],
            }
            for row in result_rows
        ]
        return session

    def add_result(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: list[dict[str, object]],
        used_local_documents: bool,
        used_web_search: bool,
        errors: list[str] | None = None,
        owner_id: str | None = None,
    ) -> dict[str, object]:
        """Save one result for an owned session and update its timestamp."""
        created_at = self._now()
        serialized_sources = json.dumps(sources)
        serialized_errors = json.dumps(errors or [])
        scope = self._scope(owner_id)

        with self._lock, self._connection:
            exists = self._connection.execute(
                "SELECT 1 FROM sessions WHERE session_id = ? AND owner_id = ?",
                (session_id, scope),
            ).fetchone()
            if not exists:
                raise SessionNotFoundError(f"Session not found: {session_id}")

            self._connection.execute(
                """
                INSERT INTO research_results
                    (session_id, question, answer, sources,
                     used_local_documents, used_web_search, errors, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    question,
                    answer,
                    serialized_sources,
                    int(used_local_documents),
                    int(used_web_search),
                    serialized_errors,
                    created_at,
                ),
            )
            self._connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (created_at, session_id),
            )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "used_local_documents": used_local_documents,
            "used_web_search": used_web_search,
            "errors": errors or [],
            "created_at": created_at,
        }

    def delete_session(
        self,
        session_id: str,
        owner_id: str | None = None,
    ) -> bool:
        """Delete an owned session and its research history."""
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM sessions WHERE session_id = ? AND owner_id = ?",
                (session_id, self._scope(owner_id)),
            )
            return cursor.rowcount > 0

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "ResearchSessionManager":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
