import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.api import app
from app.research.research_router import INSUFFICIENT_INFORMATION


class OwnershipApiTestCase(unittest.TestCase):
    """Shared setup for authenticated, isolated temporary-database API tests."""

    API_KEYS = "owner-a:key-a,owner-b:key-b"
    A_HEADERS = {"X-API-Key": "key-a"}
    B_HEADERS = {"X-API-Key": "key-b"}
    A_SECRET_DOC = "alpha-secret.txt"
    A_SECRET_TEXT = "The confidential merger codename is Project Falcon."
    B_PUBLIC_DOC = "beta-note.txt"
    B_PUBLIC_TEXT = "The public product name is Blue Sky."

    def setUp(self) -> None:
        self.env_patch = patch.dict(os.environ, {"API_KEYS": self.API_KEYS})
        self.env_patch.start()

        self.documents_dir = tempfile.TemporaryDirectory()
        self.vectors_dir = tempfile.TemporaryDirectory()
        self.sessions_dir = tempfile.TemporaryDirectory()
        self.metrics_dir = tempfile.TemporaryDirectory()
        self.audit_dir = tempfile.TemporaryDirectory()

        self.path_patches = [
            patch("app.api.DOCUMENT_DIRECTORY", Path(self.documents_dir.name)),
            patch("app.api.VECTOR_STORE_PATH", Path(self.vectors_dir.name) / "vectors"),
            patch(
                "app.api.SESSION_STORE_PATH",
                Path(self.sessions_dir.name) / "sessions.db",
            ),
            patch(
                "app.api.METRICS_STORE_PATH",
                Path(self.metrics_dir.name) / "metrics.db",
            ),
            patch(
                "app.api.AUDIT_STORE_PATH",
                Path(self.audit_dir.name) / "audit.db",
            ),
        ]
        for path_patch in self.path_patches:
            path_patch.start()

    def tearDown(self) -> None:
        for path_patch in self.path_patches:
            path_patch.stop()
        self.documents_dir.cleanup()
        self.vectors_dir.cleanup()
        self.sessions_dir.cleanup()
        self.metrics_dir.cleanup()
        self.audit_dir.cleanup()
        self.env_patch.stop()

    def _upload(self, client: TestClient, headers: dict, filename: str, text: str):
        return client.post(
            "/documents/upload",
            headers=headers,
            files={"file": (filename, text.encode("utf-8"), "text/plain")},
        )

    @staticmethod
    def _fixed_embedding() -> list[float]:
        return [1.0, 1.0]


class AuthenticationTests(OwnershipApiTestCase):
    def test_missing_credentials_are_rejected(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app) as client:
            for path in ("/documents", "/sessions", "/metrics", "/audit"):
                response = client.get(path)
                self.assertEqual(response.status_code, 401, path)
                self.assertEqual(response.json()["detail"], "Authentication required.")

    def test_invalid_credentials_are_rejected(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app) as client:
            response = client.get("/documents", headers={"X-API-Key": "wrong-key"})

        self.assertEqual(response.status_code, 401)

    def test_valid_credentials_are_accepted(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app) as client:
            response = client.get("/documents", headers=self.A_HEADERS)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["documents"], [])

    def test_health_and_root_remain_public(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app) as client:
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.get("/health").status_code, 200)

    def test_missing_configuration_fails_closed(self) -> None:
        with patch.dict(os.environ, {"API_KEYS": ""}), patch(
            "app.api.process_directory_tree", return_value=[]
        ), TestClient(app) as client:
            for path in ("/documents", "/sessions", "/metrics", "/audit"):
                response = client.get(path, headers=self.A_HEADERS)
                self.assertEqual(response.status_code, 401, path)

    def test_malformed_configuration_fails_closed(self) -> None:
        with patch.dict(os.environ, {"API_KEYS": "garbage,no-colon,:owner"}), patch(
            "app.api.process_directory_tree", return_value=[]
        ), TestClient(app) as client:
            response = client.get("/documents", headers={"X-API-Key": "key-a"})

        self.assertEqual(response.status_code, 401)

    def test_ambiguous_secret_fails_closed(self) -> None:
        with patch.dict(
            os.environ, {"API_KEYS": "owner-a:shared,owner-b:shared"}
        ), patch("app.api.process_directory_tree", return_value=[]), TestClient(
            app
        ) as client:
            response = client.get("/documents", headers={"X-API-Key": "shared"})

        self.assertEqual(response.status_code, 401)

    def test_repeated_unauthenticated_requests_write_one_audit_event(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app) as client:
            for _ in range(5):
                self.assertEqual(client.get("/documents").status_code, 401)
            events = client.app.state.audit_log.list_events(actor="anonymous")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["action"], "access_denied")

    def test_denied_audit_events_do_not_contain_the_supplied_key(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app) as client:
            client.get("/documents", headers={"X-API-Key": "wrong-secret-value"})
            events = client.app.state.audit_log.list_events(actor="anonymous")

        self.assertTrue(events)
        self.assertNotIn("wrong-secret-value", str(events))


class OwnerDirectoryTests(unittest.TestCase):
    def test_reserved_owner_ids_cannot_escape_the_document_directory(self) -> None:
        import app.api as api

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("app.api.DOCUMENT_DIRECTORY", root):
                for owner in ("..", ".", "", "../etc", "a/../../b"):
                    path = api._owner_directory(owner)
                    self.assertEqual(path.parent, root, owner)


class DocumentOwnershipTests(OwnershipApiTestCase):
    def test_owner_sees_only_own_documents(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), patch(
            "app.ingestion.document_manager.create_embedding",
            return_value=self._fixed_embedding(),
        ), TestClient(app) as client:
            self._upload(client, self.A_HEADERS, self.A_SECRET_DOC, self.A_SECRET_TEXT)
            self._upload(client, self.B_HEADERS, self.B_PUBLIC_DOC, self.B_PUBLIC_TEXT)

            a_documents = client.get("/documents", headers=self.A_HEADERS).json()["documents"]
            b_documents = client.get("/documents", headers=self.B_HEADERS).json()["documents"]

        self.assertEqual([d["filename"] for d in a_documents], [self.A_SECRET_DOC])
        self.assertEqual([d["filename"] for d in b_documents], [self.B_PUBLIC_DOC])

    def test_cross_user_document_deletion_is_denied(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), patch(
            "app.ingestion.document_manager.create_embedding",
            return_value=self._fixed_embedding(),
        ), TestClient(app) as client:
            upload = self._upload(
                client, self.A_HEADERS, self.A_SECRET_DOC, self.A_SECRET_TEXT
            )
            document_id = upload.json()["document_id"]

            denied = client.delete(f"/documents/{document_id}", headers=self.B_HEADERS)
            still_there = client.get("/documents", headers=self.A_HEADERS).json()["documents"]

        self.assertEqual(denied.status_code, 404)
        self.assertEqual(denied.json()["detail"], "Document not found.")
        self.assertEqual([d["document_id"] for d in still_there], [document_id])

    def test_owner_can_delete_own_document_and_file(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), patch(
            "app.ingestion.document_manager.create_embedding",
            return_value=self._fixed_embedding(),
        ), TestClient(app) as client:
            upload = self._upload(
                client, self.A_HEADERS, self.A_SECRET_DOC, self.A_SECRET_TEXT
            )
            document_id = upload.json()["document_id"]
            stored_file = Path(self.documents_dir.name) / "owner-a" / self.A_SECRET_DOC
            self.assertTrue(stored_file.exists())

            response = client.delete(f"/documents/{document_id}", headers=self.A_HEADERS)

        self.assertEqual(response.status_code, 200)
        self.assertFalse(stored_file.exists())

    def test_delete_failure_does_not_resurrect_document(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), patch(
            "app.ingestion.document_manager.create_embedding",
            return_value=self._fixed_embedding(),
        ), TestClient(app) as client:
            upload = self._upload(
                client, self.A_HEADERS, self.A_SECRET_DOC, self.A_SECRET_TEXT
            )
            document_id = upload.json()["document_id"]

            with patch("pathlib.Path.unlink", side_effect=OSError("denied")):
                response = client.delete(
                    f"/documents/{document_id}", headers=self.A_HEADERS
                )
            listed = client.get("/documents", headers=self.A_HEADERS).json()["documents"]

        self.assertEqual(response.status_code, 503)
        self.assertEqual([d["document_id"] for d in listed], [document_id])

    def test_failed_file_write_rolls_back_new_index(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), patch(
            "app.ingestion.document_manager.create_embedding",
            return_value=self._fixed_embedding(),
        ), TestClient(app) as client:
            owner_directory = Path(self.documents_dir.name) / "owner-a"
            owner_directory.mkdir(parents=True, exist_ok=True)
            (owner_directory / self.A_SECRET_DOC).mkdir()

            response = self._upload(
                client, self.A_HEADERS, self.A_SECRET_DOC, self.A_SECRET_TEXT
            )
            listed = client.get("/documents", headers=self.A_HEADERS).json()["documents"]

        self.assertEqual(response.status_code, 400)
        self.assertEqual(listed, [])


class SessionOwnershipTests(OwnershipApiTestCase):
    def test_sessions_are_isolated_between_owners(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app) as client:
            a_session = client.post(
                "/sessions", headers=self.A_HEADERS, json={"title": "A secret"}
            ).json()["session_id"]
            client.post("/sessions", headers=self.B_HEADERS, json={"title": "B only"})

            a_list = client.get("/sessions", headers=self.A_HEADERS).json()["sessions"]
            b_list = client.get("/sessions", headers=self.B_HEADERS).json()["sessions"]

        self.assertEqual([s["session_id"] for s in a_list], [a_session])
        self.assertEqual([s["title"] for s in b_list], ["B only"])

    def test_cross_user_session_access_is_denied(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app) as client:
            a_session = client.post(
                "/sessions", headers=self.A_HEADERS, json={"title": "A"}
            ).json()["session_id"]

            read = client.get(f"/sessions/{a_session}", headers=self.B_HEADERS)
            deleted = client.delete(f"/sessions/{a_session}", headers=self.B_HEADERS)
            researched = client.post(
                f"/sessions/{a_session}/research",
                headers=self.B_HEADERS,
                json={"question": "Anything"},
            )
            owner_read = client.get(f"/sessions/{a_session}", headers=self.A_HEADERS)

        self.assertEqual(read.status_code, 404)
        self.assertEqual(deleted.status_code, 404)
        self.assertEqual(researched.status_code, 404)
        self.assertEqual(owner_read.status_code, 200)


class RetrievalIsolationTests(OwnershipApiTestCase):
    def _run_research(self, client: TestClient, headers: dict, question: str):
        with patch(
            "app.research.research_router.create_query_embedding",
            return_value=self._fixed_embedding(),
        ), patch(
            "app.research.research_router.generate_answer",
            return_value="grounded answer",
        ) as generate_answer:
            client.app.state.research_router.web_searcher = Mock()
            client.app.state.research_router.web_searcher.search.return_value = []
            response = client.post(
                "/research", headers=headers, json={"question": question}
            )
        return response, generate_answer

    def test_retrieval_is_owner_scoped_before_generation(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), patch(
            "app.ingestion.document_manager.create_embedding",
            return_value=self._fixed_embedding(),
        ), TestClient(app) as client:
            self._upload(client, self.A_HEADERS, self.A_SECRET_DOC, self.A_SECRET_TEXT)
            self._upload(client, self.B_HEADERS, self.B_PUBLIC_DOC, self.B_PUBLIC_TEXT)

            response, generate_answer = self._run_research(
                client, self.B_HEADERS, "What is the confidential merger codename?"
            )

        self.assertEqual(response.status_code, 200)
        sources = response.json()["sources"]
        self.assertEqual([s["source"] for s in sources], [self.B_PUBLIC_DOC])
        evidence_text = str(generate_answer.call_args.args[1])
        self.assertNotIn("Project Falcon", evidence_text)
        self.assertNotIn(self.A_SECRET_DOC, evidence_text)

    def test_owner_retrieval_returns_own_document_evidence(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), patch(
            "app.ingestion.document_manager.create_embedding",
            return_value=self._fixed_embedding(),
        ), TestClient(app) as client:
            self._upload(client, self.A_HEADERS, self.A_SECRET_DOC, self.A_SECRET_TEXT)

            response, generate_answer = self._run_research(
                client, self.A_HEADERS, "What is the confidential merger codename?"
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["used_local_documents"])
        evidence_text = str(generate_answer.call_args.args[1])
        self.assertIn(self.A_SECRET_DOC, evidence_text)

    def test_deleted_documents_are_no_longer_retrievable(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), patch(
            "app.ingestion.document_manager.create_embedding",
            return_value=self._fixed_embedding(),
        ), TestClient(app) as client:
            upload = self._upload(
                client, self.A_HEADERS, self.A_SECRET_DOC, self.A_SECRET_TEXT
            )
            document_id = upload.json()["document_id"]

            before, _ = self._run_research(
                client, self.A_HEADERS, "What is the confidential merger codename?"
            )
            client.delete(f"/documents/{document_id}", headers=self.A_HEADERS)
            after, generate_answer = self._run_research(
                client, self.A_HEADERS, "What is the confidential merger codename?"
            )

        self.assertTrue(before.json()["used_local_documents"])
        self.assertEqual(after.json()["answer"], INSUFFICIENT_INFORMATION)
        self.assertFalse(after.json()["used_local_documents"])
        self.assertFalse(after.json()["sources"])
        generate_answer.assert_not_called()


class CitationTests(OwnershipApiTestCase):
    def test_citations_link_to_retrieved_passages(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), patch(
            "app.ingestion.document_manager.create_embedding",
            return_value=self._fixed_embedding(),
        ), patch(
            "app.research.research_router.create_query_embedding",
            return_value=self._fixed_embedding(),
        ), patch(
            "app.research.research_router.generate_answer",
            return_value="grounded answer",
        ), TestClient(app) as client:
            self._upload(client, self.A_HEADERS, self.A_SECRET_DOC, self.A_SECRET_TEXT)
            client.app.state.research_router.web_searcher = Mock()
            client.app.state.research_router.web_searcher.search.return_value = []
            response = client.post(
                "/research",
                headers=self.A_HEADERS,
                json={"question": "codename"},
            )

        sources = response.json()["sources"]
        self.assertEqual([s["citation_index"] for s in sources], [1])
        self.assertTrue(sources[0]["chunk_id"])
        self.assertEqual(sources[0]["source"], self.A_SECRET_DOC)

    def test_insufficient_evidence_is_stated_without_sources(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), patch(
            "app.research.research_router.create_query_embedding",
            return_value=self._fixed_embedding(),
        ), TestClient(app) as client:
            client.app.state.research_router.web_searcher = Mock()
            client.app.state.research_router.web_searcher.search.return_value = []
            response = client.post(
                "/research",
                headers=self.A_HEADERS,
                json={"question": "An unrelated question with no evidence."},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], INSUFFICIENT_INFORMATION)
        self.assertEqual(response.json()["sources"], [])


class AuditTests(OwnershipApiTestCase):
    def test_security_events_are_recorded_for_the_actor(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), patch(
            "app.ingestion.document_manager.create_embedding",
            return_value=self._fixed_embedding(),
        ), TestClient(app) as client:
            upload = self._upload(
                client, self.A_HEADERS, self.A_SECRET_DOC, self.A_SECRET_TEXT
            )
            document_id = upload.json()["document_id"]
            client.post("/sessions", headers=self.A_HEADERS, json={"title": "Audit"})
            client.delete(f"/documents/{document_id}", headers=self.A_HEADERS)

            events = client.get("/audit", headers=self.A_HEADERS).json()["events"]

        actions = [event["action"] for event in events]
        self.assertIn("document.upload", actions)
        self.assertIn("document.delete", actions)
        self.assertIn("session.create", actions)
        for event in events:
            self.assertTrue(event["event_id"])
            self.assertEqual(event["actor"], "owner-a")
            self.assertTrue(event["created_at"])
            self.assertNotIn("key-a", str(event))
            self.assertNotIn("Project Falcon", str(event))

    def test_owner_cannot_read_another_actors_audit_events(self) -> None:
        with patch("app.api.process_directory_tree", return_value=[]), TestClient(app) as client:
            client.post("/sessions", headers=self.A_HEADERS, json={"title": "A"})
            b_events = client.get("/audit", headers=self.B_HEADERS).json()["events"]

        self.assertEqual(b_events, [])


if __name__ == "__main__":
    unittest.main()
