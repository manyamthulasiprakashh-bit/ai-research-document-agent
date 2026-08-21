import os
import time
from dataclasses import dataclass, field

from app.generation.answer_generator import generate_answer
from app.research.web_search import WebSearchError, WebSearcher
from app.retrieval.embeddings import create_query_embedding
from app.retrieval.vector_store import VectorStore

INSUFFICIENT_INFORMATION = (
    "There is not enough reliable information in the available local documents "
    "or web sources to answer this question."
)


@dataclass
class ResearchResult:
    """The routing decision, evidence, and answer for a research question."""

    question: str
    routing_decision: str
    local_results: list[dict[str, str | float]] = field(default_factory=list)
    web_results: list[dict[str, str]] = field(default_factory=list)
    web_search_used: bool = False
    answer: str = ""
    errors: list[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


class ResearchRouter:
    """Route questions between local RAG evidence and web source collection."""

    def __init__(
        self,
        vector_store: VectorStore,
        web_searcher: WebSearcher | None = None,
        relevance_threshold: float | None = None,
        local_top_k: int = 2,
        web_max_results: int = 5,
    ) -> None:
        """Initialize a router with the existing vector store and web searcher."""
        self.vector_store = vector_store
        self.web_searcher = web_searcher or WebSearcher()
        self.relevance_threshold = self._load_threshold(relevance_threshold)
        self.local_top_k = local_top_k
        self.web_max_results = web_max_results

    @staticmethod
    def _load_threshold(value: float | None) -> float:
        """Load and validate the local relevance threshold."""
        if value is not None:
            threshold = value
        else:
            raw_threshold = os.getenv("RAG_RELEVANCE_THRESHOLD", "0.75")
            try:
                threshold = float(raw_threshold)
            except ValueError as error:
                raise ValueError(
                    "RAG_RELEVANCE_THRESHOLD must be a number between 0 and 1."
                ) from error

        if not 0 <= threshold <= 1:
            raise ValueError("RAG_RELEVANCE_THRESHOLD must be between 0 and 1.")

        return threshold

    def research_question(self, question: str) -> ResearchResult:
        """Retrieve evidence, optionally search the web, and generate an answer."""
        if not question.strip():
            raise ValueError("question cannot be empty")

        total_start = time.perf_counter()
        retrieval_start = time.perf_counter()
        errors: list[str] = []
        try:
            query_embedding = create_query_embedding(question)
            local_results = self.vector_store.search(
                query_embedding,
                top_k=self.local_top_k,
            )
        except (RuntimeError, ValueError) as error:
            local_results = []
            errors.append(str(error))
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        best_score = self._best_score(local_results)
        use_web = best_score < self.relevance_threshold
        web_results: list[dict[str, str]] = []

        if use_web:
            try:
                web_results = self.web_searcher.search(
                    question,
                    max_results=self.web_max_results,
                )
            except WebSearchError as error:
                errors.append(str(error))

        evidence = self._build_evidence(local_results, web_results)
        generation_start = time.perf_counter()
        if not evidence:
            answer = INSUFFICIENT_INFORMATION
            generation_latency_ms = 0.0
        else:
            try:
                answer = generate_answer(question, evidence)
            except Exception as error:
                errors.append(str(error))
                answer = INSUFFICIENT_INFORMATION
            generation_latency_ms = (time.perf_counter() - generation_start) * 1000

        return ResearchResult(
            question=question,
            routing_decision=self._routing_decision(
                best_score=best_score,
                use_web=use_web,
                web_results=web_results,
            ),
            local_results=local_results,
            web_results=web_results,
            web_search_used=use_web,
            answer=answer,
            errors=errors,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=(time.perf_counter() - total_start) * 1000,
        )

    @staticmethod
    def _best_score(results: list[dict[str, str | float]]) -> float:
        """Return the strongest local result score, or zero when there are none."""
        if not results:
            return 0.0
        return max(float(result.get("score", 0.0)) for result in results)

    @staticmethod
    def _build_evidence(
        local_results: list[dict[str, str | float]],
        web_results: list[dict[str, str]],
    ) -> list[dict[str, str | float]]:
        """Normalize local and web records for the existing answer generator."""
        evidence: list[dict[str, str | float]] = []

        for result in local_results:
            evidence.append(
                {
                    "chunk_id": result.get("chunk_id", "unknown"),
                    "source": f"LOCAL DOCUMENT: {result.get('source', 'unknown')}",
                    "text": result.get("text", ""),
                    "score": result.get("score", 0.0),
                }
            )

        for index, result in enumerate(web_results, start=1):
            evidence.append(
                {
                    "chunk_id": f"web-source-{index}",
                    "source": (
                        f"WEB SOURCE: {result.get('domain', 'unknown')} "
                        f"({result.get('url', '')})"
                    ),
                    "text": (
                        f"Title: {result.get('title', '')}\n"
                        f"Snippet: {result.get('snippet', '')}"
                    ),
                }
            )

        return evidence

    def _routing_decision(
        self,
        best_score: float,
        use_web: bool,
        web_results: list[dict[str, str]],
    ) -> str:
        """Describe why local or web evidence was selected."""
        if not use_web:
            return (
                f"local documents (best relevance {best_score:.3f} >= "
                f"threshold {self.relevance_threshold:.3f})"
            )
        if web_results:
            return (
                f"local + web sources (best local relevance {best_score:.3f} < "
                f"threshold {self.relevance_threshold:.3f})"
            )
        return (
            f"web attempted, unavailable (best local relevance {best_score:.3f} < "
            f"threshold {self.relevance_threshold:.3f})"
        )
