import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.generation import answer_generator
from app.ingestion.document_manager import discover_documents, process_documents
from app.retrieval.vector_store import VectorStore


class DocumentManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        directory = Path(self.temp_directory.name)
        (directory / "alpha.txt").write_text(
            "Retrieval systems find relevant research passages.",
            encoding="utf-8",
        )
        (directory / "beta.txt").write_text(
            "Document workflows preserve source filenames for citation.",
            encoding="utf-8",
        )
        (directory / "ignored.md").write_text(
            "This file format is not supported.",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_discovers_multiple_supported_documents(self) -> None:
        documents = discover_documents(self.temp_directory.name)

        self.assertEqual([document.name for document in documents], ["alpha.txt", "beta.txt"])

    @patch("app.ingestion.document_manager.create_embedding")
    def test_processes_documents_and_preserves_sources(self, create_embedding) -> None:
        create_embedding.return_value = [1.0, 0.0]
        vector_store = VectorStore()

        processed = process_documents(
            self.temp_directory.name,
            vector_store,
            chunk_size=50,
            overlap=5,
        )
        results = vector_store.search([1.0, 0.0], top_k=10)

        self.assertEqual(processed, ["alpha.txt", "beta.txt"])
        self.assertEqual({result["source"] for result in results}, {"alpha.txt", "beta.txt"})
        self.assertEqual(create_embedding.call_count, 2)

    @patch("app.ingestion.document_manager.create_embedding")
    def test_retrieves_chunks_from_different_sources(self, create_embedding) -> None:
        def embed(text: str) -> list[float]:
            if "research" in text.lower():
                return [1.0, 0.0]
            return [0.0, 1.0]

        create_embedding.side_effect = embed
        vector_store = VectorStore()
        process_documents(
            self.temp_directory.name,
            vector_store,
            chunk_size=50,
            overlap=5,
        )

        results = vector_store.search([1.0, 1.0], top_k=2)

        self.assertEqual(
            {result["source"] for result in results},
            {"alpha.txt", "beta.txt"},
        )
        self.assertTrue(all("score" in result for result in results))

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-only-placeholder"})
    def test_unanswered_question_prompt_requires_fallback(self) -> None:
        prompts: list[str] = []

        class FakeModels:
            def generate_content(self, *, model: str, contents: str):
                prompts.append(contents)
                return SimpleNamespace(text="The available documents do not provide enough information to answer this question.")

        class FakeClient:
            def __init__(self, *, api_key: str):
                self.models = FakeModels()

        with patch.object(answer_generator.genai, "Client", FakeClient):
            answer_generator.generate_answer(
                "What are the latest developments in quantum computing?",
                [
                    {
                        "chunk_id": "alpha.txt:chunk-1",
                        "source": "alpha.txt",
                        "text": "Retrieval systems find relevant research passages.",
                    }
                ],
            )

        self.assertIn("Do not invent facts", prompts[0])
        self.assertIn(
            "The available documents do not provide enough information",
            prompts[0],
        )
        context = prompts[0].split("Context:\n", 1)[1].split(
            "\n---\n\nQuestion:",
            1,
        )[0]
        self.assertNotIn("quantum computing", context)


if __name__ == "__main__":
    unittest.main()
