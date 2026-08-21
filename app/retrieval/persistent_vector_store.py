import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


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
    """SQLite-backed vector store for single-machine document retrieval."""

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

    @staticmethod
    def fingerprint(content: bytes) -> str:
        """Return a stable SHA-256 fingerprint for document contents."""
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def document_id_for(fingerprint: str) -> str:
        """Return a stable document ID derived from a content fingerprint."""
        return f"doc-{fingerprint}"

    def _create_tables(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    fingerprint TEXT NOT NULL UNIQUE,
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

    def add_document(
        self,
        document_id: str,
        filename: str,
        fingerprint: str,
        chunks: list[tuple[str, str, list[float]]],
    ) -> dict[str, object]:
        """Add a document and all chunks atomically, skipping duplicates."""
        now = datetime.now(timezone.utc).isoformat()

        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT document_id, filename FROM documents WHERE fingerprint = ?",
                (fingerprint,),
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
                    (document_id, filename, fingerprint, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (document_id, filename, fingerprint, now, now),
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
    ) -> None:
        """Add one chunk for compatibility with the in-memory store API."""
        resolved_document_id = document_id or self.document_id_for(
            self.fingerprint(source.encode())
        )
        fingerprint = self.fingerprint(source.encode())
        self.add_document(
            resolved_document_id,
            source,
            fingerprint,
            [(chunk_id, text, embedding)],
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
    ) -> list[dict[str, str | float]]:
        """Return the top matching chunks from persistent storage."""
        if top_k <= 0:
            return []

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT chunk_id, source, text, embedding
                FROM chunks
                """
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

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks, returning whether it existed."""
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )
            return cursor.rowcount > 0

    def list_documents(self) -> list[dict[str, str | int]]:
        """Return indexed document metadata and chunk counts."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT d.document_id, d.filename, d.fingerprint,
                       d.created_at, d.updated_at, COUNT(c.chunk_id) AS chunks
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.document_id
                GROUP BY d.document_id
                ORDER BY d.filename COLLATE NOCASE
                """
            ).fetchall()

        return [dict(row) for row in rows]

    def get_document_by_fingerprint(
        self,
        fingerprint: str,
    ) -> dict[str, str | int] | None:
        """Return indexed document metadata for a fingerprint, if present."""
        with self._lock:
            row = self._connection.execute(
                """
                SELECT d.document_id, d.filename, d.fingerprint,
                       d.created_at, d.updated_at, COUNT(c.chunk_id) AS chunks
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.document_id
                WHERE d.fingerprint = ?
                GROUP BY d.document_id
                """,
                (fingerprint,),
            ).fetchone()
        return dict(row) if row else None

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
