import inspect
import unittest
from unittest.mock import Mock, patch

from app.research import research_router
from app.research.research_router import (
    INSUFFICIENT_INFORMATION,
    ResearchRouter,
)
from app.research.web_search import WebSearchError


class ResearchRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.question = "What is RAG?"
        self.local_result = {
            "chunk_id": "sample.txt:chunk-1",
            "source": "sample.txt",
            "text": "RAG retrieves relevant context before generation.",
            "score": 0.9,
        }
        self.web_result = {
            "title": "RAG guide",
            "url": "https://example.com/rag",
            "snippet": "A guide to retrieval-augmented generation.",
            "domain": "example.com",
        }

    @patch("app.research.research_router.generate_answer", return_value="local answer")
    @patch("app.research.research_router.create_query_embedding", return_value=[1.0])
    def test_high_local_relevance_skips_web_search(
        self,
        create_query_embedding,
        generate_answer,
    ) -> None:
        vector_store = Mock()
        vector_store.search.return_value = [self.local_result]
        web_searcher = Mock()
        router = ResearchRouter(vector_store, web_searcher, relevance_threshold=0.75)

        result = router.research_question(self.question)

        web_searcher.search.assert_not_called()
        self.assertFalse(result.web_search_used)
        self.assertIn("local documents", result.routing_decision)
        generate_answer.assert_called_once()
        create_query_embedding.assert_called_once_with(self.question)

    @patch("app.research.research_router.generate_answer", return_value="combined answer")
    @patch("app.research.research_router.create_query_embedding", return_value=[1.0])
    def test_low_local_relevance_triggers_web_search(
        self,
        create_query_embedding,
        generate_answer,
    ) -> None:
        vector_store = Mock()
        low_score = {**self.local_result, "score": 0.2}
        vector_store.search.return_value = [low_score]
        web_searcher = Mock()
        web_searcher.search.return_value = [self.web_result]
        router = ResearchRouter(vector_store, web_searcher, relevance_threshold=0.75)

        result = router.research_question(self.question)

        web_searcher.search.assert_called_once_with(self.question, max_results=5)
        self.assertTrue(result.web_search_used)
        self.assertEqual(result.web_results, [self.web_result])
        self.assertIn("local + web sources", result.routing_decision)
        evidence = generate_answer.call_args.args[1]
        self.assertIn("LOCAL DOCUMENT: sample.txt", evidence[0]["source"])
        self.assertIn("WEB SOURCE: example.com", evidence[1]["source"])
        create_query_embedding.assert_called_once_with(self.question)

    @patch("app.research.research_router.generate_answer", return_value="combined answer")
    @patch("app.research.research_router.create_query_embedding", return_value=[1.0])
    def test_local_and_web_context_are_included(
        self,
        create_query_embedding,
        generate_answer,
    ) -> None:
        vector_store = Mock()
        vector_store.search.return_value = [{**self.local_result, "score": 0.4}]
        web_searcher = Mock()
        web_searcher.search.return_value = [self.web_result]
        router = ResearchRouter(vector_store, web_searcher, relevance_threshold=0.75)

        router.research_question(self.question)

        evidence = generate_answer.call_args.args[1]
        evidence_text = " ".join(str(item) for item in evidence)
        self.assertIn("RAG retrieves relevant context", evidence_text)
        self.assertIn("A guide to retrieval-augmented generation", evidence_text)

    @patch("app.research.research_router.generate_answer", return_value="web answer")
    @patch("app.research.research_router.create_query_embedding", return_value=[1.0])
    def test_empty_local_results_trigger_web_search(
        self,
        create_query_embedding,
        generate_answer,
    ) -> None:
        vector_store = Mock()
        vector_store.search.return_value = []
        web_searcher = Mock()
        web_searcher.search.return_value = [self.web_result]
        router = ResearchRouter(vector_store, web_searcher)

        result = router.research_question(self.question)

        web_searcher.search.assert_called_once()
        self.assertEqual(result.local_results, [])
        self.assertEqual(result.web_results, [self.web_result])
        generate_answer.assert_called_once()
        create_query_embedding.assert_called_once_with(self.question)

    @patch("app.research.research_router.generate_answer", return_value="local fallback")
    @patch("app.research.research_router.create_query_embedding", return_value=[1.0])
    def test_web_search_failure_falls_back_to_local_context(
        self,
        create_query_embedding,
        generate_answer,
    ) -> None:
        vector_store = Mock()
        vector_store.search.return_value = [{**self.local_result, "score": 0.2}]
        web_searcher = Mock()
        web_searcher.search.side_effect = WebSearchError("provider unavailable")
        router = ResearchRouter(vector_store, web_searcher)

        result = router.research_question(self.question)

        self.assertEqual(result.answer, "local fallback")
        self.assertEqual(result.errors, ["provider unavailable"])
        self.assertIn("web attempted, unavailable", result.routing_decision)
        generate_answer.assert_called_once()
        create_query_embedding.assert_called_once_with(self.question)

    @patch("app.research.research_router.create_query_embedding", return_value=[1.0])
    def test_no_useful_evidence_returns_insufficient_information(
        self,
        create_query_embedding,
    ) -> None:
        vector_store = Mock()
        vector_store.search.return_value = []
        web_searcher = Mock()
        web_searcher.search.return_value = []
        router = ResearchRouter(vector_store, web_searcher)

        result = router.research_question(self.question)

        self.assertEqual(result.answer, INSUFFICIENT_INFORMATION)
        self.assertEqual(result.web_results, [])
        create_query_embedding.assert_called_once_with(self.question)

    def test_api_keys_are_not_embedded_in_router_source_or_output(self) -> None:
        source = inspect.getsource(research_router)

        self.assertNotIn("test-only-placeholder", source)
        self.assertNotIn("AIza", source)
        self.assertNotIn("tvly-", source)


if __name__ == "__main__":
    unittest.main()
