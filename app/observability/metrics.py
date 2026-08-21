import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.observability.models import ResearchMetrics


class MetricsStore:
    """Persist research metrics and calculate aggregate observability values."""

    def __init__(self, path: str | Path = "data/storage/metrics.db") -> None:
        """Open or create the metrics database."""
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
                CREATE TABLE IF NOT EXISTS research_metrics (
                    request_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    question TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    total_latency_ms REAL NOT NULL,
                    retrieval_latency_ms REAL NOT NULL,
                    generation_latency_ms REAL NOT NULL,
                    retrieved_chunks INTEGER NOT NULL,
                    top_retrieval_score REAL NOT NULL,
                    average_retrieval_score REAL NOT NULL,
                    source_count INTEGER NOT NULL,
                    local_sources_count INTEGER NOT NULL,
                    web_sources_count INTEGER NOT NULL,
                    answer_length INTEGER NOT NULL,
                    citation_count INTEGER NOT NULL,
                    used_local_documents INTEGER NOT NULL,
                    used_web_search INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    error_message TEXT NOT NULL
                )
                """
            )

    def record(self, metrics: ResearchMetrics) -> None:
        """Persist one metrics record in a transaction."""
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO research_metrics (
                    request_id, session_id, question, started_at, completed_at,
                    total_latency_ms, retrieval_latency_ms, generation_latency_ms,
                    retrieved_chunks, top_retrieval_score, average_retrieval_score,
                    source_count, local_sources_count, web_sources_count,
                    answer_length, citation_count, used_local_documents,
                    used_web_search, success, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metrics.request_id,
                    metrics.session_id,
                    metrics.question,
                    metrics.started_at,
                    metrics.completed_at,
                    metrics.total_latency_ms,
                    metrics.retrieval_latency_ms,
                    metrics.generation_latency_ms,
                    metrics.retrieved_chunks,
                    metrics.top_retrieval_score,
                    metrics.average_retrieval_score,
                    metrics.source_count,
                    metrics.local_sources_count,
                    metrics.web_sources_count,
                    metrics.answer_length,
                    metrics.citation_count,
                    int(metrics.used_local_documents),
                    int(metrics.used_web_search),
                    int(metrics.success),
                    metrics.error_message,
                ),
            )

    def aggregate(self) -> dict[str, int | float]:
        """Return aggregate metrics for all recorded research requests."""
        with self._lock:
            row = self._connection.execute(
                """
                SELECT
                    COUNT(*) AS total_requests,
                    COALESCE(SUM(success), 0) AS successful_requests,
                    COALESCE(SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END), 0)
                        AS failed_requests,
                    COALESCE(AVG(total_latency_ms), 0) AS average_latency_ms,
                    COALESCE(AVG(retrieval_latency_ms), 0)
                        AS average_retrieval_latency_ms,
                    COALESCE(AVG(generation_latency_ms), 0)
                        AS average_generation_latency_ms,
                    COALESCE(AVG(top_retrieval_score), 0)
                        AS average_retrieval_score,
                    COALESCE(AVG(source_count), 0) AS average_sources
                FROM research_metrics
                """
            ).fetchone()
        return {
            key: round(float(value), 3) if isinstance(value, float) else int(value)
            for key, value in dict(row).items()
        }

    def close(self) -> None:
        """Close the metrics database connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "MetricsStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def citation_count(answer: str, sources: list[dict[str, Any]]) -> int:
    """Count sources represented by their supplied identifiers in an answer."""
    answer_lower = answer.lower()
    count = 0
    for source in sources:
        candidates = [
            source.get("source"),
            source.get("url"),
            source.get("title"),
        ]
        if any(
            isinstance(candidate, str)
            and candidate.strip()
            and candidate.lower() in answer_lower
            for candidate in candidates
        ):
            count += 1
    return count


def metrics_from_result(
    result: Any,
    request_id: str,
    session_id: str | None,
    started_at: str,
    completed_at: str,
    total_latency_ms: float,
    error_message: str = "",
) -> ResearchMetrics:
    """Build metrics from the existing ResearchResult contract."""
    local_scores = [
        float(item.get("score", 0.0))
        for item in result.local_results
    ]
    sources = [
        {
            "source_type": "local_document",
            "source": item.get("source", ""),
            "chunk_id": item.get("chunk_id", ""),
            "score": item.get("score", 0.0),
        }
        for item in result.local_results
    ] + [
        {
            "source_type": "web",
            "source": item.get("domain", ""),
            "title": item.get("title", ""),
            "url": item.get("url", ""),
        }
        for item in result.web_results
    ]
    return ResearchMetrics(
        request_id=request_id,
        session_id=session_id,
        question=result.question,
        started_at=started_at,
        completed_at=completed_at,
        total_latency_ms=total_latency_ms,
        retrieval_latency_ms=float(getattr(result, "retrieval_latency_ms", 0.0)),
        generation_latency_ms=float(getattr(result, "generation_latency_ms", 0.0)),
        retrieved_chunks=len(result.local_results),
        top_retrieval_score=max(local_scores, default=0.0),
        average_retrieval_score=(sum(local_scores) / len(local_scores)) if local_scores else 0.0,
        source_count=len(sources),
        local_sources_count=len(result.local_results),
        web_sources_count=len(result.web_results),
        answer_length=len(result.answer),
        citation_count=citation_count(result.answer, sources),
        used_local_documents=bool(result.local_results),
        used_web_search=result.web_search_used,
        success=not bool(result.errors) and not bool(error_message),
        error_message=error_message or "; ".join(result.errors),
    )


def now_iso() -> str:
    """Return the current UTC timestamp for request metrics."""
    return datetime.now(timezone.utc).isoformat()
