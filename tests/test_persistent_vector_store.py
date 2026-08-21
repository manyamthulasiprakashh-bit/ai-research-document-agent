import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.ingestion.document_manager import process_document
from app.retrieval.persistent_vector_store import PersistentVectorStore


class PersistentVectorStoreTests(unittest.TestCase):
    def test_index_survives_store_recreation_and_is_searchable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document_path = root / "research.txt"
            document_path.write_text(
                "Retrieval-Augmented Generation retrieves relevant context.",
                encoding="utf-8",
            )

            store = PersistentVectorStore(root / "vectors")
            with patch(
                "app.ingestion.document_manager.create_embedding",
                return_value=[1.0, 0.0],
            ) as create_embedding:
                result = process_document(document_path, store, chunk_size=50, overlap=5)

            self.assertEqual(result["status"], "indexed")
            self.assertEqual(store.count(), 1)
            store.save()
            store.close()

            recreated_store = PersistentVectorStore.load(root / "vectors")
            results = recreated_store.search([1.0, 0.0], top_k=1)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["source"], "research.txt")
            self.assertEqual(results[0]["text"], "Retrieval-Augmented Generation retrieves relevant context.")
            self.assertEqual(create_embedding.call_count, 1)
            recreated_store.close()

    def test_duplicate_content_is_not_indexed_twice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.txt"
            second = root / "renamed.txt"
            first.write_text("The same document content.", encoding="utf-8")
            second.write_text("The same document content.", encoding="utf-8")
            store = PersistentVectorStore(root / "vectors")

            with patch(
                "app.ingestion.document_manager.create_embedding",
                return_value=[0.0, 1.0],
            ) as create_embedding:
                first_result = process_document(first, store, chunk_size=50, overlap=5)
                second_result = process_document(second, store, chunk_size=50, overlap=5)

            self.assertEqual(first_result["status"], "indexed")
            self.assertEqual(second_result["status"], "duplicate")
            self.assertEqual(first_result["document_id"], second_result["document_id"])
            self.assertEqual(store.count(), 1)
            self.assertEqual(create_embedding.call_count, 1)
            store.close()

    def test_list_documents_reports_chunk_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PersistentVectorStore(directory)
            result = store.add_document(
                document_id="doc-one",
                filename="notes.txt",
                fingerprint="fingerprint-one",
                chunks=[
                    ("notes.txt:chunk-1", "one", [1.0, 0.0]),
                    ("notes.txt:chunk-2", "two", [0.0, 1.0]),
                ],
            )

            documents = store.list_documents()

            self.assertEqual(result["status"], "indexed")
            self.assertEqual(documents[0]["document_id"], "doc-one")
            self.assertEqual(documents[0]["filename"], "notes.txt")
            self.assertEqual(documents[0]["chunks"], 2)
            store.close()

    def test_delete_document_removes_document_and_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PersistentVectorStore(directory)
            store.add_document(
                document_id="doc-delete",
                filename="delete.txt",
                fingerprint="fingerprint-delete",
                chunks=[("delete.txt:chunk-1", "remove me", [1.0, 0.0])],
            )

            self.assertTrue(store.delete_document("doc-delete"))
            self.assertFalse(store.list_documents())
            self.assertEqual(store.count(), 0)
            self.assertFalse(store.delete_document("doc-delete"))
            store.close()


if __name__ == "__main__":
    unittest.main()
