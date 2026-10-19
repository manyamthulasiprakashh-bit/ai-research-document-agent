import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.ingestion.document_manager import process_document
from app.research.session_manager import ResearchSessionManager, SessionNotFoundError
from app.retrieval.persistent_vector_store import PersistentVectorStore


def _create_legacy_vector_database(path: Path) -> None:
    """Create a pre-ownership vector database schema with one document."""
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE documents (
            document_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            fingerprint TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents(document_id)
                ON DELETE CASCADE,
            source TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
        ("doc-old", "legacy.txt", "abc123", "2026-01-01", "2026-01-01"),
    )
    connection.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
        (
            "legacy.txt:chunk-1",
            "doc-old",
            "legacy.txt",
            "legacy content",
            json.dumps([1.0, 0.0]),
            "2026-01-01",
        ),
    )
    connection.commit()
    connection.close()


def _create_legacy_session_database(path: Path) -> None:
    """Create a pre-ownership session database schema with one session."""
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE research_results (
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
        """
    )
    connection.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?)",
        ("session-old", "Legacy Session", "2026-01-01", "2026-01-01"),
    )
    connection.commit()
    connection.close()


class VectorStoreMigrationTests(unittest.TestCase):
    def test_legacy_document_is_preserved_and_owner_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "vectors.db"
            _create_legacy_vector_database(database_path)

            store = PersistentVectorStore(database_path)

            migrated = store.get_document("doc-old", owner_id="legacy")
            self.assertIsNotNone(migrated)
            self.assertEqual(migrated["document_id"], "doc-old")
            self.assertEqual(migrated["filename"], "legacy.txt")
            self.assertEqual(migrated["fingerprint"], "abc123")
            self.assertEqual(migrated["chunks"], 1)

            legacy_results = store.search([1.0, 0.0], owner_id="legacy")
            self.assertEqual(len(legacy_results), 1)
            self.assertEqual(legacy_results[0]["text"], "legacy content")

            self.assertEqual(store.search([1.0, 0.0], owner_id="owner-a"), [])
            self.assertIsNone(store.get_document("doc-old", owner_id="owner-a"))
            self.assertEqual(store.count(), 1)
            store.close()

    def test_identical_content_is_isolated_per_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document_path = root / "shared.txt"
            document_path.write_text("Identical content for both owners.", encoding="utf-8")
            store = PersistentVectorStore(root / "vectors")

            with patch(
                "app.ingestion.document_manager.create_embedding",
                return_value=[1.0, 0.0],
            ):
                owner_a = process_document(
                    document_path, store, chunk_size=50, overlap=5, owner_id="owner-a"
                )
                owner_b = process_document(
                    document_path, store, chunk_size=50, overlap=5, owner_id="owner-b"
                )

            self.assertNotEqual(owner_a["document_id"], owner_b["document_id"])
            self.assertEqual(store.count(), 2)
            self.assertEqual(len(store.list_documents(owner_id="owner-a")), 1)
            self.assertEqual(len(store.list_documents(owner_id="owner-b")), 1)
            self.assertEqual(len(store.search([1.0, 0.0], owner_id="owner-a")), 1)
            store.close()


class SessionMigrationTests(unittest.TestCase):
    def test_legacy_session_is_preserved_and_owner_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "sessions.db"
            _create_legacy_session_database(database_path)

            manager = ResearchSessionManager(database_path)

            migrated = manager.get_session("session-old", owner_id="legacy")
            self.assertEqual(migrated["title"], "Legacy Session")
            self.assertEqual(migrated["questions"], [])

            with self.assertRaises(SessionNotFoundError):
                manager.get_session("session-old", owner_id="owner-a")

            self.assertEqual(manager.list_sessions(owner_id="legacy")[0]["session_id"], "session-old")
            self.assertEqual(manager.list_sessions(owner_id="owner-a"), [])
            manager.close()


if __name__ == "__main__":
    unittest.main()
