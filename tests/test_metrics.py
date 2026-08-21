import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.observability.metrics import MetricsStore, citation_count, metrics_from_result
from app.observability.models import ResearchMetrics


class MetricsTests(unittest.TestCase):
    def test_metrics_creation_and_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MetricsStore(Path(directory) / "metrics.db")
            metrics = ResearchMetrics(
                request_id="request-1",
                session_id=None,
                question="What is RAG?",
                started_at="2026-01-01T00:00:00+00:00",
                completed_at="2026-01-01T00:00:01+00:00",
                total_latency_ms=1000.0,
                retrieval_latency_ms=120.0,
                generation_latency_ms=800.0,
                retrieved_chunks=2,
                top_retrieval_score=0.9,
                average_retrieval_score=0.8,
                source_count=2,
                local_sources_count=2,
                web_sources_count=0,
                answer_length=42,
                citation_count=1,
                used_local_documents=True,
                used_web_search=False,
                success=True,
            )
            store.record(metrics)
            aggregate = store.aggregate()
            store.close()

        self.assertEqual(aggregate["total_requests"], 1)
        self.assertEqual(aggregate["successful_requests"], 1)
        self.assertEqual(aggregate["failed_requests"], 0)
        self.assertEqual(aggregate["average_latency_ms"], 1000.0)
        self.assertEqual(aggregate["average_retrieval_score"], 0.9)

    def test_citation_count_is_deterministic(self) -> None:
        sources = [
            {"source": "research_notes.txt"},
            {"source": "ibm.com", "url": "https://ibm.com/rag", "title": "RAG"},
        ]

        self.assertEqual(
            citation_count("See research_notes.txt and https://ibm.com/rag.", sources),
            2,
        )
        self.assertEqual(citation_count("No source cited.", sources), 0)

    def test_metrics_from_empty_result_uses_safe_zero_values(self) -> None:
        result = SimpleNamespace(
            question="Empty",
            local_results=[],
            web_results=[],
            answer="",
            web_search_used=False,
            errors=[],
            retrieval_latency_ms=0.0,
            generation_latency_ms=0.0,
        )

        metrics = metrics_from_result(
            result,
            "request-empty",
            None,
            "start",
            "complete",
            0.0,
        )

        self.assertEqual(metrics.retrieved_chunks, 0)
        self.assertEqual(metrics.top_retrieval_score, 0.0)
        self.assertEqual(metrics.average_retrieval_score, 0.0)
        self.assertEqual(metrics.source_count, 0)
        self.assertEqual(metrics.citation_count, 0)

    def test_failed_metrics_are_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MetricsStore(Path(directory) / "metrics.db")
            store.record(
                ResearchMetrics(
                    request_id="failed",
                    session_id=None,
                    question="failed",
                    started_at="start",
                    completed_at="complete",
                    total_latency_ms=4.0,
                    retrieval_latency_ms=0.0,
                    generation_latency_ms=0.0,
                    retrieved_chunks=0,
                    top_retrieval_score=0.0,
                    average_retrieval_score=0.0,
                    source_count=0,
                    local_sources_count=0,
                    web_sources_count=0,
                    answer_length=0,
                    citation_count=0,
                    used_local_documents=False,
                    used_web_search=False,
                    success=False,
                    error_message="failure",
                )
            )
            aggregate = store.aggregate()
            store.close()

        self.assertEqual(aggregate["failed_requests"], 1)
        self.assertEqual(aggregate["successful_requests"], 0)


if __name__ == "__main__":
    unittest.main()
