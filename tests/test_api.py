import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import app
from app.research.research_router import ResearchResult


class ApiTests(unittest.TestCase):
    API_KEYS = "owner-a:key-a,owner-b:key-b"
    OWNER_A = "owner-a"
    OWNER_B = "owner-b"
    KEY_A = "key-a"
    KEY_B = "key-b"
    API_HEADERS = {"X-API-Key": "key-a"}

    def setUp(self) -> None:
        self.env_patch = patch.dict(os.environ, {"API_KEYS": self.API_KEYS})
        self.env_patch.start()
        self.test_storage = tempfile.TemporaryDirectory()
        self.storage_patch = patch("app.api.VECTOR_STORE_PATH", Path(self.test_storage.name))
        self.storage_patch.start()
        self.session_storage = tempfile.TemporaryDirectory()
        self.session_storage_patch = patch(
            "app.api.SESSION_STORE_PATH",
            Path(self.session_storage.name) / "sessions.db",
        )
        self.session_storage_patch.start()
        self.metrics_storage = tempfile.TemporaryDirectory()
        self.metrics_storage_patch = patch(
            "app.api.METRICS_STORE_PATH",
            Path(self.metrics_storage.name) / "metrics.db",
        )
        self.metrics_storage_patch.start()
        self.audit_storage = tempfile.TemporaryDirectory()
        self.audit_storage_patch = patch(
            "app.api.AUDIT_STORE_PATH",
            Path(self.audit_storage.name) / "audit.db",
        )
        self.audit_storage_patch.start()

    def tearDown(self) -> None:
        self.storage_patch.stop()
        self.test_storage.cleanup()
        self.session_storage_patch.stop()
        self.session_storage.cleanup()
        self.metrics_storage_patch.stop()
        self.metrics_storage.cleanup()
        self.audit_storage_patch.stop()
        self.audit_storage.cleanup()
        self.env_patch.stop()

    def test_root_health_and_openapi(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app, headers=self.API_HEADERS) as client:
            root_response = client.get("/")
            health_response = client.get("/health")
            docs_response = client.get("/docs")
            schema_response = client.get("/openapi.json")

        self.assertEqual(root_response.status_code, 200)
        self.assertEqual(root_response.json()["status"], "ok")
        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(health_response.json()["status"], "ok")
        self.assertEqual(docs_response.status_code, 200)
        self.assertEqual(schema_response.status_code, 200)
        self.assertIn("/research", schema_response.json()["paths"])

    def test_ui_returns_complete_page_and_assets(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app, headers=self.API_HEADERS) as client:
            page_response = client.get("/ui")
            css_response = client.get("/ui/assets/styles.css")
            javascript_response = client.get("/ui/assets/app.js")

        self.assertEqual(page_response.status_code, 200)
        self.assertIn("AI Research &amp; Document Agent", page_response.text)
        self.assertIn("Ask a research question", page_response.text)
        self.assertIn(">Ask Research Question<", page_response.text)
        self.assertEqual(css_response.status_code, 200)
        self.assertEqual(javascript_response.status_code, 200)
        self.assertIn('requestWithRetry("/sessions")', javascript_response.text)
        self.assertNotIn('http://127.0.0.1:8000/sessions', javascript_response.text)

    def test_health_remains_available_when_ingestion_fails(self) -> None:
        with patch(
            "app.api.process_directory_tree",
            side_effect=RuntimeError("GEMINI_API_KEY is missing"),
        ), TestClient(app, headers=self.API_HEADERS) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "initialization": "degraded"})

    def test_research_with_valid_question(self) -> None:
        fake_result = ResearchResult(
            question="What is RAG?",
            routing_decision="local documents",
            local_results=[
                {
                    "source": "sample.txt",
                    "chunk_id": "sample.txt:chunk-1",
                    "score": 0.9,
                    "text": "RAG retrieves context.",
                }
            ],
            answer="RAG retrieves context before generation.",
        )
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app, headers=self.API_HEADERS) as client:
            with patch.object(
                client.app.state.research_router,
                "research_question",
                return_value=fake_result,
            ) as research_question:
                response = client.post(
                    "/research",
                    json={"question": "What is RAG?"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], fake_result.answer)
        self.assertTrue(response.json()["used_local_documents"])
        self.assertFalse(response.json()["used_web_search"])
        self.assertEqual(response.json()["sources"][0]["source"], "sample.txt")
        research_question.assert_called_once_with("What is RAG?", owner_id=self.OWNER_A)

    def test_empty_question_is_rejected(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app, headers=self.API_HEADERS) as client:
            response = client.post("/research", json={"question": "   "})

        self.assertEqual(response.status_code, 422)

    def test_research_router_error_is_handled(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app, headers=self.API_HEADERS) as client:
            with patch.object(
                client.app.state.research_router,
                "research_question",
                side_effect=RuntimeError("Gemini unavailable"),
            ):
                response = client.post(
                    "/research",
                    json={"question": "What is RAG?"},
                )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Research service is temporarily unavailable.")

    def test_research_errors_are_sanitized_for_clients(self) -> None:
        fake_result = ResearchResult(
            question="What is RAG?",
            routing_decision="web search",
            answer="An answer.",
            errors=[
                "GEMINI_API_KEY is missing",
                "provider exploded at https://internal.example/trace",
            ],
        )
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(
            app, headers=self.API_HEADERS
        ) as client:
            with patch.object(
                client.app.state.research_router,
                "research_question",
                return_value=fake_result,
            ):
                response = client.post(
                    "/research",
                    json={"question": "What is RAG?"},
                )

        self.assertEqual(response.status_code, 200)
        errors = response.json()["errors"]
        self.assertTrue(errors)
        joined = " ".join(errors)
        self.assertNotIn("GEMINI_API_KEY", joined)
        self.assertNotIn("internal.example", joined)

    def test_upload_rejects_unsupported_file_type(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app, headers=self.API_HEADERS) as client:
            response = client.post(
                "/documents/upload",
                files={"file": ("notes.exe", b"not supported", "application/octet-stream")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Only .txt, .pdf, and .docx", response.json()["detail"])

    def test_upload_processes_supported_file_through_document_manager(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document_directory = Path(directory) / "documents"
            with patch("app.api.DOCUMENT_DIRECTORY", document_directory), patch(
                "app.api.process_directory_tree", return_value=["uploaded.txt"]
            ), patch(
                "app.api.process_document",
                return_value={
                    "document_id": "doc-uploaded",
                    "filename": "uploaded.txt",
                    "chunks_created": 1,
                    "status": "indexed",
                },
            ) as process_document, TestClient(app, headers=self.API_HEADERS) as client:
                response = client.post(
                    "/documents/upload",
                    files={"file": ("uploaded.txt", b"A research note.", "text/plain")},
                )

            self.assertEqual(response.status_code, 200)
            self.assertTrue((document_directory / self.OWNER_A / "uploaded.txt").exists())
            self.assertEqual(response.json()["filename"], "uploaded.txt")
            self.assertEqual(response.json()["document_id"], "doc-uploaded")
            self.assertEqual(response.json()["chunks_created"], 1)
            process_document.assert_called_once()

    def test_documents_lists_persistent_metadata(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app, headers=self.API_HEADERS) as client:
            client.app.state.vector_store.add_document(
                document_id="doc-list",
                filename="listed.txt",
                fingerprint="fingerprint-list",
                chunks=[("listed.txt:chunk-1", "content", [1.0, 0.0])],
                owner_id=self.OWNER_A,
            )
            response = client.get("/documents")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["documents"][0]["document_id"], "doc-list")
        self.assertEqual(response.json()["documents"][0]["chunks"], 1)

    def test_documents_delete_removes_persistent_metadata(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app, headers=self.API_HEADERS) as client:
            client.app.state.vector_store.add_document(
                document_id="doc-delete-api",
                filename="delete.txt",
                fingerprint="fingerprint-delete-api",
                chunks=[("delete.txt:chunk-1", "content", [1.0, 0.0])],
                owner_id=self.OWNER_A,
            )
            response = client.delete("/documents/doc-delete-api")
            missing_response = client.delete("/documents/doc-delete-api")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "deleted")
        self.assertEqual(missing_response.status_code, 404)

    def test_create_and_list_sessions(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app, headers=self.API_HEADERS) as client:
            create_response = client.post("/sessions", json={"title": "RAG Research"})
            list_response = client.get("/sessions")

        self.assertEqual(create_response.status_code, 200)
        created = create_response.json()
        self.assertEqual(created["title"], "RAG Research")
        self.assertTrue(created["session_id"])
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(list_response.json()["sessions"][0]["session_id"], created["session_id"])

    def test_session_research_and_history(self) -> None:
        fake_result = ResearchResult(
            question="What is RAG?",
            routing_decision="local documents",
            local_results=[
                {
                    "source": "research_notes.txt",
                    "chunk_id": "research_notes.txt:chunk-1",
                    "score": 0.91,
                    "text": "RAG retrieves context.",
                }
            ],
            answer="RAG retrieves context before generation.",
        )
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app, headers=self.API_HEADERS) as client:
            session_id = client.post("/sessions", json={"title": "RAG Research"}).json()["session_id"]
            with patch.object(
                client.app.state.research_router,
                "research_question",
                return_value=fake_result,
            ) as research_question:
                research_response = client.post(
                    f"/sessions/{session_id}/research",
                    json={"question": "What is RAG?"},
                )
            detail_response = client.get(f"/sessions/{session_id}")

        self.assertEqual(research_response.status_code, 200)
        self.assertEqual(research_response.json()["created_at"], detail_response.json()["questions"][0]["created_at"])
        self.assertEqual(detail_response.status_code, 200)
        self.assertEqual(detail_response.json()["title"], "RAG Research")
        self.assertEqual(detail_response.json()["questions"][0]["answer"], fake_result.answer)
        research_question.assert_called_once_with("What is RAG?", owner_id=self.OWNER_A)

    def test_session_delete_and_invalid_session(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app, headers=self.API_HEADERS) as client:
            session_id = client.post("/sessions", json={"title": "Temporary"}).json()["session_id"]
            delete_response = client.delete(f"/sessions/{session_id}")
            detail_response = client.get(f"/sessions/{session_id}")
            research_response = client.post(
                f"/sessions/{session_id}/research",
                json={"question": "Question"},
            )

        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(detail_response.status_code, 404)
        self.assertEqual(research_response.status_code, 404)

    def test_session_empty_title_and_question_are_rejected(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app, headers=self.API_HEADERS) as client:
            empty_title = client.post("/sessions", json={"title": "   "})
            session_id = client.post("/sessions", json={"title": "Valid"}).json()["session_id"]
            empty_question = client.post(
                f"/sessions/{session_id}/research",
                json={"question": "   "},
            )

        self.assertEqual(empty_title.status_code, 422)
        self.assertEqual(empty_question.status_code, 422)

    def test_metrics_endpoint_returns_aggregates(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app, headers=self.API_HEADERS) as client:
            response = client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total_requests"], 0)
        self.assertEqual(response.json()["successful_requests"], 0)

    def test_research_records_metrics(self) -> None:
        fake_result = ResearchResult(
            question="What is RAG?",
            routing_decision="local documents",
            local_results=[
                {
                    "source": "research_notes.txt",
                    "chunk_id": "research_notes.txt:chunk-1",
                    "score": 0.9,
                    "text": "retrieval and generation",
                }
            ],
            answer="research_notes.txt explains retrieval and generation.",
        )
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app, headers=self.API_HEADERS) as client:
            with patch.object(
                client.app.state.research_router,
                "research_question",
                return_value=fake_result,
            ):
                response = client.post("/research", json={"question": "What is RAG?"})
            metrics_response = client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(metrics_response.status_code, 200)
        self.assertEqual(metrics_response.json()["total_requests"], 1)
        self.assertEqual(metrics_response.json()["successful_requests"], 1)
        self.assertEqual(metrics_response.json()["average_sources"], 1.0)

    def test_evaluation_endpoint_uses_mocked_router(self) -> None:
        fake_result = ResearchResult(
            question="Question",
            routing_decision="local documents",
            local_results=[{"source": "research_notes.txt", "score": 0.9}],
            answer="retrieval generation research_notes.txt",
        )
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app, headers=self.API_HEADERS) as client:
            with patch.object(
                client.app.state.research_router,
                "research_question",
                return_value=fake_result,
            ) as research_question:
                response = client.post("/evaluation/run")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total_questions"], 10)
        self.assertEqual(response.json()["retrieval_hit_rate"], 0.3)
        self.assertEqual(research_question.call_count, 10)


if __name__ == "__main__":
    unittest.main()
