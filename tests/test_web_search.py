import unittest
from unittest.mock import Mock, patch

import requests

from app.research.web_search import WebSearchError, WebSearcher


class WebSearcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.searcher = WebSearcher(api_key="test-only-placeholder")
        self.provider_results = [
            {
                "title": "RAG overview",
                "url": "https://example.com/rag",
                "content": "An overview of retrieval-augmented generation.",
            },
            {
                "title": "Research paper",
                "url": "https://research.example.org/paper",
                "snippet": "A research paper about RAG.",
            },
        ]

    @patch("app.research.web_search.requests.post")
    def test_search_returns_structured_results(self, post) -> None:
        response = Mock()
        response.json.return_value = {"results": self.provider_results}
        post.return_value = response

        results = self.searcher.search("What is RAG?")

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["title"], "RAG overview")
        self.assertEqual(results[0]["snippet"], "An overview of retrieval-augmented generation.")
        self.assertEqual(results[0]["domain"], "example.com")

    @patch("app.research.web_search.requests.post")
    def test_max_results_is_respected(self, post) -> None:
        response = Mock()
        response.json.return_value = {"results": self.provider_results * 2}
        post.return_value = response

        results = self.searcher.search("What is RAG?", max_results=1)

        self.assertEqual(len(results), 1)
        request_payload = post.call_args.kwargs["json"]
        self.assertEqual(request_payload["max_results"], 1)

    @patch("app.research.web_search.requests.post")
    def test_empty_results_are_handled(self, post) -> None:
        response = Mock()
        response.json.return_value = {"results": []}
        post.return_value = response

        self.assertEqual(self.searcher.search("unknown topic"), [])

    @patch("app.research.web_search.requests.post")
    def test_provider_errors_raise_clean_error(self, post) -> None:
        post.side_effect = requests.RequestException("provider unavailable")

        with self.assertRaises(WebSearchError) as error:
            self.searcher.search("What is RAG?")

        self.assertEqual(str(error.exception), "The web search provider request failed.")
        self.assertNotIn("test-only-placeholder", str(error.exception))

    @patch("app.research.web_search.requests.post")
    def test_every_result_contains_required_fields(self, post) -> None:
        response = Mock()
        response.json.return_value = {"results": self.provider_results}
        post.return_value = response

        results = self.searcher.search("What is RAG?")

        for result in results:
            self.assertEqual(
                set(result),
                {"title", "url", "snippet", "domain"},
            )
            self.assertTrue(all(isinstance(value, str) for value in result.values()))


if __name__ == "__main__":
    unittest.main()
