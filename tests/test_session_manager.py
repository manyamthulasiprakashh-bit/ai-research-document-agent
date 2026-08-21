import tempfile
import unittest
from pathlib import Path

from app.research.session_manager import ResearchSessionManager, SessionNotFoundError


class ResearchSessionManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.manager = ResearchSessionManager(Path(self.directory.name) / "sessions.db")

    def tearDown(self) -> None:
        self.manager.close()
        self.directory.cleanup()

    def test_create_and_retrieve_session(self) -> None:
        created = self.manager.create_session("RAG Research")
        session = self.manager.get_session(created["session_id"])

        self.assertEqual(session["title"], "RAG Research")
        self.assertEqual(session["created_at"], created["created_at"])
        self.assertEqual(session["questions"], [])

    def test_list_sessions(self) -> None:
        first = self.manager.create_session("First")
        second = self.manager.create_session("Second")

        sessions = self.manager.list_sessions()

        session_ids = {session["session_id"] for session in sessions}
        self.assertEqual(session_ids, {first["session_id"], second["session_id"]})

    def test_add_result_and_retrieve_history(self) -> None:
        session = self.manager.create_session("RAG Research")
        result = self.manager.add_result(
            session_id=session["session_id"],
            question="What is RAG?",
            answer="RAG retrieves context before generation.",
            sources=[
                {
                    "source_type": "local_document",
                    "source": "research_notes.txt",
                    "score": 0.9,
                }
            ],
            used_local_documents=True,
            used_web_search=False,
        )

        history = self.manager.get_session(session["session_id"])

        self.assertEqual(result["question"], "What is RAG?")
        self.assertEqual(len(history["questions"]), 1)
        self.assertEqual(history["questions"][0]["answer"], result["answer"])
        self.assertTrue(history["questions"][0]["used_local_documents"])
        self.assertEqual(history["questions"][0]["sources"][0]["source"], "research_notes.txt")

    def test_result_updates_session_timestamp(self) -> None:
        session = self.manager.create_session("Timestamp Test")
        original_updated_at = self.manager.get_session(session["session_id"])["updated_at"]

        self.manager.add_result(
            session_id=session["session_id"],
            question="Question",
            answer="Answer",
            sources=[],
            used_local_documents=False,
            used_web_search=False,
        )
        updated_session = self.manager.get_session(session["session_id"])

        self.assertGreater(updated_session["updated_at"], original_updated_at)

    def test_delete_session_removes_history(self) -> None:
        session = self.manager.create_session("Delete Me")
        self.manager.add_result(
            session_id=session["session_id"],
            question="Question",
            answer="Answer",
            sources=[],
            used_local_documents=False,
            used_web_search=False,
        )

        self.assertTrue(self.manager.delete_session(session["session_id"]))
        self.assertFalse(self.manager.list_sessions())
        with self.assertRaises(SessionNotFoundError):
            self.manager.get_session(session["session_id"])

    def test_multiple_sessions_remain_isolated(self) -> None:
        first = self.manager.create_session("First")
        second = self.manager.create_session("Second")
        self.manager.add_result(
            session_id=first["session_id"],
            question="First question",
            answer="First answer",
            sources=[],
            used_local_documents=False,
            used_web_search=False,
        )

        first_history = self.manager.get_session(first["session_id"])
        second_history = self.manager.get_session(second["session_id"])

        self.assertEqual(len(first_history["questions"]), 1)
        self.assertEqual(len(second_history["questions"]), 0)

    def test_missing_session_result_is_rejected(self) -> None:
        with self.assertRaises(SessionNotFoundError):
            self.manager.add_result(
                session_id="missing",
                question="Question",
                answer="Answer",
                sources=[],
                used_local_documents=False,
                used_web_search=False,
            )


if __name__ == "__main__":
    unittest.main()
