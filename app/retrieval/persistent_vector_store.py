import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PUBLIC_SCOPE = "public"
LEGACY_SCOPE = "legacy"


@dataclass(frozen=True)
class IndexedDocument:
    """Metadata for one document in the persistent knowledge base."""

    document_id: str
    filename: str
    fingerprint: str
    chunks: int
    created_at: str
    updated_at: str


class PersistentVectorStore:
    """SQLite-backed vector store for single-machine document retrieval.

    Documents are owned by exactly one owner. Retrieval, listing, and deletion
    are scoped to an owner so one owner can never observe another owner's
    document metadata, chunks, or embeddings.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        """Open or create a persistent vector store database."""
        configured_path = path or "data/vector_store"
        store_path = Path(configured_path)
        if store_path.suffix == ".db":
            self._database_path = store_path
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            store_path.mkdir(parents=True, exist_ok=True)
            self._database_path = store_path / "vectors.db"

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
    def fingerprint(content: bytes) -> str:
        """Return a stable SHA-256 fingerprint for document contents."""
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _scope(owner_id: str | None) -> str:
        """Return the normalized owner scope for an optional owner id."""
        return owner_id or PUBLIC_SCOPE

    @classmethod
    def fingerprint_key(cls, fingerprint: str, owner_id: str | None = None) -> str:
        """Return an owner-scoped storage key so dedup is per owner."""
        return f"{cls._scope(owner_id)}:{fingerprint}"

    @classmethod
    def document_id_for(
        cls,
        fingerprint: str,
        owner_id: str | None = None,
    ) -> str:
        """Return an owner-scoped stable document ID for a content fingerprint."""
        scope = cls._scope(owner_id)
        digest = hashlib.sha256(f"{scope}:{fingerprint}".encode()).hexdigest()
        return f"doc-{digest}"

    @staticmethod
    def _strip_scope(stored_key: str) -> str:
        """Return the raw content fingerprint from an owner-scoped key."""
        return stored_key.split(":", 1)[1] if ":" in stored_key else stored_key

    def _create_tables(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    fingerprint TEXT NOT NULL UNIQUE,
                    owner_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(document_id)
                        ON DELETE CASCADE,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                    ON chunks(document_id);
                """
            )

    def _migrate(self) -> None:
        """Add ownership to databases created before this feature.

        The migration only adds a column and rewrites the stored fingerprint to
        its owner-scoped form. Existing document IDs, chunks, and embeddings are
        preserved. Rows created before ownership existed are assigned to the
        legacy owner.
        """
        with self._lock, self._connection:
            columns = {
                row["name"]
                for row in self._connection.execute(
                    "PRAGMA table_info(documents)"
                ).fetchall()
            }
            if "owner_id" in columns:
                return

            self._connection.execute(
                "ALTER TABLE documents ADD COLUMN owner_id TEXT"
            )
            self._connection.execute(
                """
                UPDATE documents
                SET owner_id = ?, fingerprint = ? || ':' || fingerprint
                WHERE owner_id IS NULL
                """,
                (LEGACY_SCOPE, LEGACY_SCOPE),
            )

    def add_document(
        self,
        document_id: str,
        filename: str,
        fingerprint: str,
        chunks: list[tuple[str, str, list[float]]],
        owner_id: str | None = None,
    ) -> dict[str, object]:
        """Add a document and all chunks atomically, skipping per-owner duplicates."""
        now = datetime.now(timezone.utc).isoformat()
        scope = self._scope(owner_id)
        stored_key = self.fingerprint_key(fingerprint, owner_id)

        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT document_id, filename FROM documents WHERE fingerprint = ?",
                (stored_key,),
            ).fetchone()
            if existing:
                existing_count = self._connection.execute(
                    "SELECT COUNT(*) FROM chunks WHERE document_id = ?",
                    (existing["document_id"],),
                ).fetchone()[0]
                return {
                    "document_id": existing["document_id"],
                    "filename": existing["filename"],
                    "chunks_created": existing_count,
                    "status": "duplicate",
                }

            self._connection.execute(
                """
                INSERT INTO documents
                    (document_id, filename, fingerprint, owner_id,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (document_id, filename, stored_key, scope, now, now),
            )
            self._connection.executemany(
                """
                INSERT INTO chunks
                    (chunk_id, document_id, source, text, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk_id,
                        document_id,
                        filename,
                        text,
                        json.dumps(embedding),
                        now,
                    )
                    for chunk_id, text, embedding in chunks
                ],
            )

        return {
            "document_id": document_id,
            "filename": filename,
            "chunks_created": len(chunks),
            "status": "indexed",
        }

    def add(
        self,
        chunk_id: str,
        text: str,
        embedding: list[float],
        source: str,
        document_id: str | None = None,
        owner_id: str | None = None,
    ) -> None:
        """Add one chunk for compatibility with the in-memory store API."""
        fingerprint = self.fingerprint(source.encode())
        resolved_document_id = document_id or self.document_id_for(
            fingerprint,
            owner_id,
        )
        self.add_document(
            resolved_document_id,
            source,
            fingerprint,
            [(chunk_id, text, embedding)],
            owner_id=owner_id,
        )

    @staticmethod
    def cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        """Return cosine similarity between two vectors."""
        if len(vector_a) != len(vector_b):
            raise ValueError("Vectors must have the same dimension.")

        magnitude_a = sum(value * value for value in vector_a) ** 0.5
        magnitude_b = sum(value * value for value in vector_b) ** 0.5
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        dot_product = sum(
            value_a * value_b for value_a, value_b in zip(vector_a, vector_b)
        )
        return dot_product / (magnitude_a * magnitude_b)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        owner_id: str | None = None,
    ) -> list[dict[str, str | float]]:
        """Return the top matching chunks owned by the requested owner.

        Owner filtering happens in SQL before any ranking, so unauthorized
        chunks are never scored or returned.
        """
        if top_k <= 0:
            return []

        scope = self._scope(owner_id)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT c.chunk_id, c.source, c.text, c.embedding
                FROM chunks c
                JOIN documents d ON c.document_id = d.document_id
                WHERE d.owner_id = ?
                """,
                (scope,),
            ).fetchall()

        results = []
        for row in rows:
            embedding = json.loads(row["embedding"])
            results.append(
                {
                    "chunk_id": row["chunk_id"],
                    "source": row["source"],
                    "text": row["text"],
                    "score": self.cosine_similarity(query_embedding, embedding),
                }
            )

        results.sort(key=lambda result: float(result["score"]), reverse=True)
        return results[:top_k]

    def get_document(
        self,
        document_id: str,
        owner_id: str | None = None,
    ) -> dict[str, str | int] | None:
        """Return one owned document's metadata, if the owner owns it."""
        scope = self._scope(owner_id)
        with self._lock:
            row = self._connection.execute(
                """
                SELECT d.document_id, d.filename, d.fingerprint, d.owner_id,
                       d.created_at, d.updated_at,
                       COUNT(c.chunk_id) AS chunks
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.document_id
                WHERE d.document_id = ? AND d.owner_id = ?
                GROUP BY d.document_id
                """,
                (document_id, scope),
            ).fetchone()
        return self._document_dict(row) if row else None

    def delete_document(
        self,
        document_id: str,
        owner_id: str | None = None,
    ) -> bool:
        """Delete an owned document and its chunks, returning whether it existed."""
        scope = self._scope(owner_id)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM documents WHERE document_id = ? AND owner_id = ?",
                (document_id, scope),
            )
            return cursor.rowcount > 0

    def list_documents(
        self,
        owner_id: str | None = None,
    ) -> list[dict[str, str | int]]:
        """Return indexed document metadata for one owner and chunk counts."""
        scope = self._scope(owner_id)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT d.document_id, d.filename, d.fingerprint, d.owner_id,
                       d.created_at, d.updated_at, COUNT(c.chunk_id) AS chunks
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.document_id
                WHERE d.owner_id = ?
                GROUP BY d.document_id
                ORDER BY d.filename COLLATE NOCASE
                """,
                (scope,),
            ).fetchall()

        return [self._document_dict(row) for row in rows]

    def get_document_by_fingerprint(
        self,
        fingerprint: str,
        owner_id: str | None = None,
    ) -> dict[str, str | int] | None:
        """Return owned document metadata for a content fingerprint, if present."""
        scope = self._scope(owner_id)
        stored_key = self.fingerprint_key(fingerprint, owner_id)
        with self._lock:
            row = self._connection.execute(
                """
                SELECT d.document_id, d.filename, d.fingerprint, d.owner_id,
                       d.created_at, d.updated_at, COUNT(c.chunk_id) AS chunks
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.document_id
                WHERE d.fingerprint = ? AND d.owner_id = ?
                GROUP BY d.document_id
                """,
                (stored_key, scope),
            ).fetchone()
        return self._document_dict(row) if row else None

    def _document_dict(self, row: sqlite3.Row) -> dict[str, str | int]:
        """Convert a joined document row and hide the internal scoped key."""
        document = dict(row)
        document["fingerprint"] = self._strip_scope(str(document["fingerprint"]))
        return document

    def count(self) -> int:
        """Return the number of stored chunks."""
        with self._lock:
            return self._connection.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]

    def save(self) -> None:
        """Commit pending SQLite work to disk."""
        with self._lock:
            self._connection.commit()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PersistentVectorStore":
        """Open an existing persistent store or create its directory."""
        return cls(path)

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "PersistentVectorStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
