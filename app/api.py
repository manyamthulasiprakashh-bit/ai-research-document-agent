from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import tempfile
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.ingestion.document_manager import (
    SUPPORTED_EXTENSIONS,
    process_document,
    process_documents,
)
from app.evaluation.dataset import EVALUATION_DATASET
from app.evaluation.evaluator import run_evaluation
from app.observability.metrics import MetricsStore, metrics_from_result, now_iso
from app.observability.models import ResearchMetrics
from app.research.research_router import ResearchRouter
from app.research.session_manager import ResearchSessionManager, SessionNotFoundError
from app.retrieval.persistent_vector_store import PersistentVectorStore

load_dotenv()

logger = logging.getLogger(__name__)

DOCUMENT_DIRECTORY = Path("data/documents")
WEB_DIRECTORY = Path(__file__).parent / "web"
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", "data/vector_store"))
SESSION_STORE_PATH = Path(
    os.getenv("SESSION_STORE_PATH", "data/storage/research_sessions.db")
)
METRICS_STORE_PATH = Path(
    os.getenv("METRICS_STORE_PATH", "data/storage/metrics.db")
)


class ResearchRequest(BaseModel):
    """Request body for a research question."""

    question: str = Field(..., min_length=1)

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        """Reject questions containing only whitespace."""
        question = value.strip()
        if not question:
            raise ValueError("question cannot be empty")
        return question


class SourceResponse(BaseModel):
    """A local-document or web source returned with an answer."""

    source_type: str
    source: str
    chunk_id: str | None = None
    score: float | None = None
    title: str | None = None
    url: str | None = None
    snippet: str | None = None


class ResearchResponse(BaseModel):
    """API response containing an answer and its source metadata."""

    question: str
    answer: str
    sources: list[SourceResponse]
    used_local_documents: bool
    used_web_search: bool
    errors: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    """Result of a validated document upload."""

    filename: str
    document_id: str
    chunks_created: int
    status: str


class DocumentResponse(BaseModel):
    """Indexed document metadata returned by the API."""

    document_id: str
    filename: str
    chunks: int


class DocumentsResponse(BaseModel):
    """Collection of indexed document metadata."""

    documents: list[DocumentResponse]


class SessionCreateRequest(BaseModel):
    """Request body for creating a research session."""

    title: str = Field(..., min_length=1)

    @field_validator("title")
    @classmethod
    def title_must_contain_text(cls, value: str) -> str:
        """Reject titles containing only whitespace."""
        title = value.strip()
        if not title:
            raise ValueError("title cannot be empty")
        return title


class SessionCreatedResponse(BaseModel):
    """Metadata returned after creating a session."""

    session_id: str
    title: str
    created_at: str


class SessionSummary(BaseModel):
    """Summary metadata for a research session."""

    session_id: str
    title: str
    created_at: str
    updated_at: str


class SessionsResponse(BaseModel):
    """Collection of research session summaries."""

    sessions: list[SessionSummary]


class SessionResearchResponse(ResearchResponse):
    """A research result persisted inside a session."""

    created_at: str


class SessionDetailResponse(SessionSummary):
    """Session metadata and its complete research history."""

    questions: list[SessionResearchResponse]


class MetricsResponse(BaseModel):
    """Aggregated research observability metrics."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency_ms: float
    average_retrieval_latency_ms: float
    average_generation_latency_ms: float
    average_retrieval_score: float
    average_sources: float


class EvaluationResponse(BaseModel):
    """Baseline deterministic RAG evaluation results."""

    total_questions: int
    retrieval_hit_rate: float
    source_match_rate: float
    keyword_coverage: float
    average_latency_ms: float


def _sources_from_result(result) -> list[SourceResponse]:
    """Convert router evidence into API source records."""
    sources: list[SourceResponse] = []

    for local_result in result.local_results:
        sources.append(
            SourceResponse(
                source_type="local_document",
                source=str(local_result.get("source", "")),
                chunk_id=str(local_result.get("chunk_id", "")),
                score=float(local_result.get("score", 0.0)),
            )
        )

    for web_result in result.web_results:
        sources.append(
            SourceResponse(
                source_type="web",
                source=str(web_result.get("domain", "")),
                title=str(web_result.get("title", "")),
                url=str(web_result.get("url", "")),
                snippet=str(web_result.get("snippet", "")),
            )
        )

    return sources


def _research_response(result) -> ResearchResponse:
    """Convert a router result into the public research response model."""
    return ResearchResponse(
        question=result.question,
        answer=result.answer,
        sources=_sources_from_result(result),
        used_local_documents=bool(result.local_results),
        used_web_search=result.web_search_used,
        errors=list(result.errors),
    )


def _session_manager(request: Request) -> ResearchSessionManager:
    """Return the initialized session manager or raise a safe API error."""
    manager = getattr(request.app.state, "session_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Research session storage is temporarily unavailable.",
        )
    return manager


def _record_metrics_safely(
    request: Request,
    metrics: ResearchMetrics,
) -> None:
    """Persist metrics without allowing observability to break research."""
    metrics_store = getattr(request.app.state, "metrics_store", None)
    if metrics_store is None:
        return
    try:
        metrics_store.record(metrics)
    except Exception:
        logger.exception("Metrics recording failed for request %s", metrics.request_id)


def _record_failed_request(
    request: Request,
    request_id: str,
    session_id: str | None,
    question: str,
    started_at: str,
    total_latency_ms: float,
    error_message: str,
) -> None:
    """Record a failed request with safe zero-valued retrieval metrics."""
    completed_at = now_iso()
    _record_metrics_safely(
        request,
        ResearchMetrics(
            request_id=request_id,
            session_id=session_id,
            question=question,
            started_at=started_at,
            completed_at=completed_at,
            total_latency_ms=total_latency_ms,
            retrieval_latency_ms=0.0,
            generation_latency_ms=0.0,
            retrieved_chunks=0,
            top_retrieval_score=0.0,
            average_retrieval_score=0.0,
            source_count=0,
            local_sources_count=0,
            web_sources_count=0,
            answer_length=0,
            citation_count=0,
            used_local_documents=False,
            used_web_search=False,
            success=False,
            error_message=error_message,
        ),
    )


def _record_result_metrics(
    request: Request,
    result,
    request_id: str,
    session_id: str | None,
    started_at: str,
    total_started: float,
) -> None:
    """Build and persist metrics for a completed router result."""
    _record_metrics_safely(
        request,
        metrics_from_result(
            result=result,
            request_id=request_id,
            session_id=session_id,
            started_at=started_at,
            completed_at=now_iso(),
            total_latency_ms=(time.perf_counter() - total_started) * 1000,
        ),
    )


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize the local store and router for the application lifetime."""
    vector_store = PersistentVectorStore(VECTOR_STORE_PATH)
    initialization_error = ""

    try:
        DOCUMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        process_documents(DOCUMENT_DIRECTORY, vector_store)
    except Exception as error:
        initialization_error = str(error)

    try:
        router = ResearchRouter(vector_store)
    except ValueError as error:
        initialization_error = str(error)
        router = ResearchRouter(vector_store, relevance_threshold=0.75)

    try:
        session_manager = ResearchSessionManager(SESSION_STORE_PATH)
        session_error = ""
    except Exception as error:
        session_manager = None
        session_error = str(error)
        initialization_error = initialization_error or session_error

    try:
        metrics_store = MetricsStore(METRICS_STORE_PATH)
        metrics_error = ""
    except Exception as error:
        metrics_store = None
        metrics_error = str(error)
        initialization_error = initialization_error or metrics_error

    application.state.vector_store = vector_store
    application.state.research_router = router
    application.state.session_manager = session_manager
    application.state.metrics_store = metrics_store
    application.state.initialization_error = initialization_error
    try:
        yield
    finally:
        vector_store.close()
        if session_manager is not None:
            session_manager.close()
        if metrics_store is not None:
            metrics_store.close()


app = FastAPI(
    title="AI Research & Document Agent",
    description="A local research API combining document RAG with web search.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount(
    "/ui/assets",
    StaticFiles(directory=WEB_DIRECTORY),
    name="ui-assets",
)


@app.get("/")
def root() -> dict[str, str]:
    """Return the service status and UI location."""
    return {
        "status": "ok",
        "service": "AI Research & Document Agent",
        "ui": "/ui",
    }


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    """Return a simple health response."""
    initialization_error = getattr(request.app.state, "initialization_error", "")
    if initialization_error:
        return {"status": "ok", "initialization": "degraded"}
    return {"status": "ok"}


@app.get("/metrics", response_model=MetricsResponse)
def metrics(request: Request) -> MetricsResponse:
    """Return aggregate metrics for recorded research requests."""
    metrics_store = getattr(request.app.state, "metrics_store", None)
    if metrics_store is None:
        raise HTTPException(status_code=503, detail="Metrics storage is unavailable.")
    try:
        return MetricsResponse(**metrics_store.aggregate())
    except Exception as error:
        raise HTTPException(status_code=503, detail="Metrics could not be loaded.") from error


@app.post("/evaluation/run", response_model=EvaluationResponse)
def evaluation(request: Request) -> EvaluationResponse:
    """Run the deterministic local RAG evaluation dataset."""
    logger.info("Evaluation started")
    router: ResearchRouter = request.app.state.research_router
    try:
        report = run_evaluation(EVALUATION_DATASET, router.research_question)
    except Exception as error:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=503, detail="Evaluation could not be completed.") from error
    logger.info("Evaluation completed: %s questions", report["total_questions"])
    return EvaluationResponse(
        total_questions=int(report["total_questions"]),
        retrieval_hit_rate=float(report["retrieval_hit_rate"]),
        source_match_rate=float(report["source_match_rate"]),
        keyword_coverage=float(report["keyword_coverage"]),
        average_latency_ms=float(report["average_latency_ms"]),
    )


@app.post("/sessions", response_model=SessionCreatedResponse)
def create_session(
    request: Request,
    payload: SessionCreateRequest,
) -> SessionCreatedResponse:
    """Create a persistent research session."""
    manager = _session_manager(request)
    try:
        session = manager.create_session(payload.title)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Research session could not be created.",
        ) from error
    return SessionCreatedResponse(**session)


@app.get("/sessions", response_model=SessionsResponse)
def list_sessions(request: Request) -> SessionsResponse:
    """List all persistent research sessions."""
    manager = _session_manager(request)
    try:
        sessions = manager.list_sessions()
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Research sessions could not be loaded.",
        ) from error
    return SessionsResponse(sessions=[SessionSummary(**session) for session in sessions])


@app.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session(request: Request, session_id: str) -> SessionDetailResponse:
    """Return one session and its complete research history."""
    manager = _session_manager(request)
    try:
        session = manager.get_session(session_id)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail="Session not found.") from error
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Research session could not be loaded.",
        ) from error
    return SessionDetailResponse(**session)


@app.delete("/sessions/{session_id}")
def delete_session(request: Request, session_id: str) -> dict[str, object]:
    """Delete a session and its persisted research history."""
    manager = _session_manager(request)
    try:
        deleted = manager.delete_session(session_id)
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Research session could not be deleted.",
        ) from error
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session_id": session_id, "status": "deleted"}


@app.get("/documents", response_model=DocumentsResponse)
def list_documents(request: Request) -> DocumentsResponse:
    """List documents currently indexed in the persistent knowledge base."""
    vector_store: PersistentVectorStore = request.app.state.vector_store
    return DocumentsResponse(
        documents=[
            DocumentResponse(
                document_id=str(document["document_id"]),
                filename=str(document["filename"]),
                chunks=int(document["chunks"]),
            )
            for document in vector_store.list_documents()
        ]
    )


@app.get("/ui", response_class=FileResponse, include_in_schema=False)
def ui() -> str:
    """Serve the local research web interface."""
    return str(WEB_DIRECTORY / "index.html")


@app.post("/research", response_model=ResearchResponse)
def research(request: Request, payload: ResearchRequest) -> ResearchResponse:
    """Research a question through the existing hybrid router."""
    router: ResearchRouter = request.app.state.research_router
    request_id = str(uuid.uuid4())
    started_at = now_iso()
    total_started = time.perf_counter()
    logger.info("Research started: request_id=%s", request_id)

    try:
        result = router.research_question(payload.question)
    except ValueError as error:
        _record_failed_request(
            request, request_id, None, payload.question, started_at,
            (time.perf_counter() - total_started) * 1000, str(error),
        )
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        _record_failed_request(
            request, request_id, None, payload.question, started_at,
            (time.perf_counter() - total_started) * 1000,
            "Research service is temporarily unavailable.",
        )
        raise HTTPException(
            status_code=503,
            detail="Research service is temporarily unavailable.",
        ) from error

    errors = list(result.errors)
    initialization_error = getattr(request.app.state, "initialization_error", "")
    if initialization_error and not result.local_results:
        errors.insert(0, initialization_error)

    response = _research_response(result)
    response.errors = errors
    _record_result_metrics(request, result, request_id, None, started_at, total_started)
    logger.info("Research completed: request_id=%s", request_id)
    return response


@app.post(
    "/sessions/{session_id}/research",
    response_model=SessionResearchResponse,
)
def research_in_session(
    request: Request,
    session_id: str,
    payload: ResearchRequest,
) -> SessionResearchResponse:
    """Research a question and persist the result in an existing session."""
    manager = _session_manager(request)
    try:
        manager.get_session(session_id)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail="Session not found.") from error
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Research session could not be loaded.",
        ) from error

    router: ResearchRouter = request.app.state.research_router
    request_id = str(uuid.uuid4())
    started_at = now_iso()
    total_started = time.perf_counter()
    logger.info("Research started: request_id=%s session_id=%s", request_id, session_id)
    try:
        result = router.research_question(payload.question)
    except ValueError as error:
        _record_failed_request(
            request, request_id, session_id, payload.question, started_at,
            (time.perf_counter() - total_started) * 1000, str(error),
        )
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        _record_failed_request(
            request, request_id, session_id, payload.question, started_at,
            (time.perf_counter() - total_started) * 1000,
            "Research service is temporarily unavailable.",
        )
        raise HTTPException(
            status_code=503,
            detail="Research service is temporarily unavailable.",
        ) from error

    errors = list(result.errors)
    initialization_error = getattr(request.app.state, "initialization_error", "")
    if initialization_error and not result.local_results:
        errors.insert(0, initialization_error)

    response = _research_response(result)
    response.errors = errors
    try:
        stored_result = manager.add_result(
            session_id=session_id,
            question=response.question,
            answer=response.answer,
            sources=[source.model_dump() for source in response.sources],
            used_local_documents=response.used_local_documents,
            used_web_search=response.used_web_search,
            errors=response.errors,
        )
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail="Session not found.") from error
    except Exception as error:
        _record_failed_request(
            request, request_id, session_id, payload.question, started_at,
            (time.perf_counter() - total_started) * 1000,
            "Research result could not be saved.",
        )
        raise HTTPException(
            status_code=503,
            detail="Research result could not be saved.",
        ) from error

    _record_result_metrics(
        request,
        result,
        request_id,
        session_id,
        started_at,
        total_started,
    )
    logger.info(
        "Research completed: request_id=%s session_id=%s",
        request_id,
        session_id,
    )
    return SessionResearchResponse(
        question=str(stored_result["question"]),
        answer=str(stored_result["answer"]),
        sources=[SourceResponse(**source) for source in stored_result["sources"]],
        used_local_documents=bool(stored_result["used_local_documents"]),
        used_web_search=bool(stored_result["used_web_search"]),
        errors=list(stored_result["errors"]),
        created_at=str(stored_result["created_at"]),
    )


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
) -> UploadResponse:
    """Validate and process one uploaded TXT, PDF, or DOCX document."""
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if not filename or extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only .txt, .pdf, and .docx files are supported.",
        )

    DOCUMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = DOCUMENT_DIRECTORY / filename
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded document is empty.")

    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory) / filename
            temporary_path.write_bytes(content)
            indexing_result = process_document(
                temporary_path,
                request.app.state.vector_store,
            )
        destination.write_bytes(content)
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded document could not be processed.",
        ) from error

    return UploadResponse(
        filename=filename,
        document_id=str(indexing_result["document_id"]),
        chunks_created=int(indexing_result["chunks_created"]),
        status=str(indexing_result["status"]),
    )


@app.delete("/documents/{document_id}")
def delete_document(request: Request, document_id: str) -> dict[str, object]:
    """Delete an indexed document and all of its chunks."""
    vector_store: PersistentVectorStore = request.app.state.vector_store
    deleted = vector_store.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"document_id": document_id, "status": "deleted"}
