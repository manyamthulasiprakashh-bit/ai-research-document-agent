import math
from dataclasses import dataclass


@dataclass
class StoredChunk:
    """A document chunk and its embedding metadata."""

    chunk_id: str
    text: str
    embedding: list[float]
    source: str


class VectorStore:
    """Store document embeddings and retrieve the most similar chunks."""

    def __init__(self) -> None:
        """Initialize an empty vector store."""
        self._chunks: list[StoredChunk] = []

    def add(
        self,
        chunk_id: str,
        text: str,
        embedding: list[float],
        source: str,
    ) -> None:
        """Add a chunk, its embedding, and its source document."""
        self._chunks.append(
            StoredChunk(
                chunk_id=chunk_id,
                text=text,
                embedding=embedding,
                source=source,
            )
        )

    @staticmethod
    def cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        """Return the cosine similarity between two vectors."""
        if len(vector_a) != len(vector_b):
            raise ValueError("Vectors must have the same dimension.")

        magnitude_a = math.sqrt(sum(value * value for value in vector_a))
        magnitude_b = math.sqrt(sum(value * value for value in vector_b))

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
        """Return the top matching chunks ranked by cosine similarity.

        This in-memory store is single-owner. The ``owner_id`` parameter is
        accepted for API compatibility with the persistent store and ignored.
        """
        if top_k <= 0:
            return []

        results = [
            {
                "score": self.cosine_similarity(query_embedding, chunk.embedding),
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
            }
            for chunk in self._chunks
        ]

        results.sort(key=lambda result: result["score"], reverse=True)
        return results[:top_k]
