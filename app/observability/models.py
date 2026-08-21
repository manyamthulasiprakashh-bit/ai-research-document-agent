from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchMetrics:
    """Measured observability data for one research request."""

    request_id: str
    session_id: str | None
    question: str
    started_at: str
    completed_at: str
    total_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    retrieved_chunks: int
    top_retrieval_score: float
    average_retrieval_score: float
    source_count: int
    local_sources_count: int
    web_sources_count: int
    answer_length: int
    citation_count: int
    used_local_documents: bool
    used_web_search: bool
    success: bool
    error_message: str = ""
