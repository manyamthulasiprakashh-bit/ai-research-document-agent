import logging

from app.evaluation.dataset import EVALUATION_DATASET
from app.evaluation.evaluator import run_evaluation
from app.ingestion.document_manager import process_documents
from app.research.research_router import ResearchRouter
from app.retrieval.persistent_vector_store import PersistentVectorStore


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    """Run the baseline evaluation against the local research pipeline."""
    store = PersistentVectorStore()
    try:
        process_documents("data/documents", store)
        router = ResearchRouter(store)
        report = run_evaluation(EVALUATION_DATASET, router.research_question)
    finally:
        store.close()

    print("====================================")
    print("RAG EVALUATION REPORT")
    print("====================================")
    print(f"Questions evaluated: {report['total_questions']}")
    print(f"Retrieval hit rate: {float(report['retrieval_hit_rate']) * 100:.1f}%")
    print(f"Source match rate: {float(report['source_match_rate']) * 100:.1f}%")
    print(f"Average keyword coverage: {float(report['keyword_coverage']) * 100:.1f}%")
    print(f"Average retrieval latency: {float(report['average_retrieval_latency_ms']):.1f} ms")
    print(f"Average generation latency: {float(report['average_generation_latency_ms']):.1f} ms")
    print(f"Average total latency: {float(report['average_latency_ms']):.1f} ms")
    print("====================================")


if __name__ == "__main__":
    main()
