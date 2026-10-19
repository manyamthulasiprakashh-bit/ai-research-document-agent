from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import tempfile
import time
import uuid

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.ingestion.document_manager import (
    SUPPORTED_EXTENSIONS,
    process_document,
    process_directory_tree,
)
from app.evaluation.dataset import EVALUATION_DATASET
from app.evaluation.evaluator import run_evaluation
from app.observability.audit import AuditLog
from app.observability.metrics import MetricsStore, metrics_from_result, now_iso
from app.observability.models import ResearchMetrics
from app.research.research_router import ResearchRouter
from app.research.session_manager import ResearchSessionManager, SessionNotFoundError
from app.retrieval.persistent_vector_store import PersistentVectorStore
from app.security.auth import (
    AccessDeniedThrottle,
    AuthenticatedUser,
    load_api_keys,
    require_user,
)

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
AUDIT_STORE_PATH = Path(
    os.getenv("AUDIT_STORE_PATH", "data/storage/audit.db")
)
LEGACY_OWNER = os.getenv("LEGACY_OWNER_ID") or "legacy"


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
    citation_index: int | None = None
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
    """Convert router evidence into API source records.

    Citation indexes map one-to-one onto the evidence actually retrieved and
    passed to the model, so no source can be cited that was not retrieved.
    """
    sources: list[SourceResponse] = []

    citation_index = 1
    for local_result in result.local_results:
        sources.append(
            SourceResponse(
                source_type="local_document",
                source=str(local_result.get("source", "")),
                citation_index=citation_index,
                chunk_id=str(local_result.get("chunk_id", "")),
                score=float(local_result.get("score", 0.0)),
            )
        )
        citation_index += 1

    for web_result in result.web_results:
        sources.append(
            SourceResponse(
                source_type="web",
                source=str(web_result.get("domain", "")),
                citation_index=citation_index,
                title=str(web_result.get("title", "")),
                url=str(web_result.get("url", "")),
                snippet=str(web_result.get("snippet", "")),
            )
        )
        citation_index += 1

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


SAFE_PROVIDER_ERROR_MESSAGE = (
    "Some retrieval, web, or generation sources were unavailable; "
    "the answer may be incomplete."
)


def _safe_client_errors(errors: list[str]) -> list[str]:
    """Return client-safe errors without internal provider or config details.

    Detailed error strings are kept out of API responses (and stored session
    results); operators can still see them in server logs.
    """
    if not errors:
        return []
    return [SAFE_PROVIDER_ERROR_MESSAGE]


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


def _owner_directory(owner_id: str) -> Path:
    """Return the per-owner upload directory, sanitized to one path segment."""
    safe_segment = "".join(
        character if character.isalnum() or character in ("-", "_", ".") else "_"
        for character in owner_id
    )
    if not safe_segment or safe_segment in (".", ".."):
        safe_segment = "owner"
    return DOCUMENT_DIRECTORY / safe_segment


def _record_audit_safely(
    request: Request,
    actor: str,
    action: str,
    resource_id: str = "",
    outcome: str = "success",
    detail: str = "",
) -> None:
    """Persist an audit event without allowing auditing to break a request."""
    audit_log = getattr(request.app.state, "audit_log", None)
    if audit_log is None:
        return
    try:
        audit_log.record(
            actor=actor,
            action=action,
            resource_id=resource_id,
            outcome=outcome,
            detail=detail,
        )
    except Exception:
        logger.exception("Audit recording failed for action %s", action)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize the local store and router for the application lifetime."""
    vector_store = PersistentVectorStore(VECTOR_STORE_PATH)
    initialization_error = ""

    try:
        DOCUMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        process_directory_tree(
            DOCUMENT_DIRECTORY,
            vector_store,
            default_owner=LEGACY_OWNER,
        )
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

    try:
        audit_log = AuditLog(AUDIT_STORE_PATH)
        audit_error = ""
    except Exception as error:
        audit_log = None
        audit_error = str(error)
        initialization_error = initialization_error or audit_error

    application.state.vector_store = vector_store
    application.state.research_router = router
    application.state.session_manager = session_manager
    application.state.metrics_store = metrics_store
    application.state.audit_log = audit_log
    application.state.api_keys = load_api_keys()
    application.state.denial_throttle = AccessDeniedThrottle()
    application.state.initialization_error = initialization_error
    try:
        yield
    finally:
        vector_store.close()
        if session_manager is not None:
            session_manager.close()
        if metrics_store is not None:
            metrics_store.close()
        if audit_log is not None:
            audit_log.close()


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
def metrics(
    request: Request,
    user: AuthenticatedUser = Depends(require_user),
) -> MetricsResponse:
    """Return aggregate metrics for recorded research requests."""
    metrics_store = getattr(request.app.state, "metrics_store", None)
    if metrics_store is None:
        raise HTTPException(status_code=503, detail="Metrics storage is unavailable.")
    try:
        return MetricsResponse(**metrics_store.aggregate())
    except Exception as error:
        raise HTTPException(status_code=503, detail="Metrics could not be loaded.") from error


@app.get("/audit")
def audit_events(
    request: Request,
    user: AuthenticatedUser = Depends(require_user),
) -> dict[str, list[dict[str, str]]]:
    """Return audit events recorded for the authenticated owner only."""
    audit_log = getattr(request.app.state, "audit_log", None)
    if audit_log is None:
        raise HTTPException(status_code=503, detail="Audit storage is unavailable.")
    try:
        return {"events": audit_log.list_events(actor=user.user_id)}
    except Exception as error:
        raise HTTPException(status_code=503, detail="Audit events could not be loaded.") from error


@app.post("/evaluation/run", response_model=EvaluationResponse)
def evaluation(
    request: Request,
    user: AuthenticatedUser = Depends(require_user),
) -> EvaluationResponse:
    """Run the deterministic local RAG evaluation dataset for this owner."""
    logger.info("Evaluation started")
    router: ResearchRouter = request.app.state.research_router
    try:
        report = run_evaluation(
            EVALUATION_DATASET,
            lambda question: router.research_question(question, owner_id=user.user_id),
        )
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
    user: AuthenticatedUser = Depends(require_user),
) -> SessionCreatedResponse:
    """Create a persistent research session owned by the authenticated user."""
    manager = _session_manager(request)
    try:
        session = manager.create_session(payload.title, owner_id=user.user_id)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Research session could not be created.",
        ) from error
    _record_audit_safely(
        request, user.user_id, "session.create", session["session_id"]
    )
    return SessionCreatedResponse(**session)


@app.get("/sessions", response_model=SessionsResponse)
def list_sessions(
    request: Request,
    user: AuthenticatedUser = Depends(require_user),
) -> SessionsResponse:
    """List the authenticated user's persistent research sessions."""
    manager = _session_manager(request)
    try:
        sessions = manager.list_sessions(owner_id=user.user_id)
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Research sessions could not be loaded.",
        ) from error
    return SessionsResponse(sessions=[SessionSummary(**session) for session in sessions])


@app.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session(
    request: Request,
    session_id: str,
    user: AuthenticatedUser = Depends(require_user),
) -> SessionDetailResponse:
    """Return one owned session and its complete research history."""
    manager = _session_manager(request)
    try:
        session = manager.get_session(session_id, owner_id=user.user_id)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail="Session not found.") from error
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Research session could not be loaded.",
        ) from error
    return SessionDetailResponse(**session)


@app.delete("/sessions/{session_id}")
def delete_session(
    request: Request,
    session_id: str,
    user: AuthenticatedUser = Depends(require_user),
) -> dict[str, object]:
    """Delete an owned session and its persisted research history."""
    manager = _session_manager(request)
    try:
        deleted = manager.delete_session(session_id, owner_id=user.user_id)
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Research session could not be deleted.",
        ) from error
    if not deleted:
        _record_audit_safely(
            request, user.user_id, "session.delete", session_id, outcome="not_found"
        )
        raise HTTPException(status_code=404, detail="Session not found.")
    _record_audit_safely(request, user.user_id, "session.delete", session_id)
    return {"session_id": session_id, "status": "deleted"}


@app.get("/documents", response_model=DocumentsResponse)
def list_documents(
    request: Request,
    user: AuthenticatedUser = Depends(require_user),
) -> DocumentsResponse:
    """List documents owned by the authenticated user."""
    vector_store: PersistentVectorStore = request.app.state.vector_store
    return DocumentsResponse(
        documents=[
            DocumentResponse(
                document_id=str(document["document_id"]),
                filename=str(document["filename"]),
                chunks=int(document["chunks"]),
            )
            for document in vector_store.list_documents(owner_id=user.user_id)
        ]
    )


@app.get("/ui", response_class=FileResponse, include_in_schema=False)
def ui() -> str:
    """Serve the local research web interface."""
    return str(WEB_DIRECTORY / "index.html")


@app.post("/research", response_model=ResearchResponse)
def research(
    request: Request,
    payload: ResearchRequest,
    user: AuthenticatedUser = Depends(require_user),
) -> ResearchResponse:
    """Research a question through the existing hybrid router."""
    router: ResearchRouter = request.app.state.research_router
    request_id = str(uuid.uuid4())
    started_at = now_iso()
    total_started = time.perf_counter()
    logger.info("Research started: request_id=%s", request_id)

    try:
        result = router.research_question(payload.question, owner_id=user.user_id)
    except ValueError as error:
        _record_failed_request(
            request, request_id, None, payload.question, started_at,
            (time.perf_counter() - total_started) * 1000, str(error),
        )
        _record_audit_safely(
            request, user.user_id, "research", request_id, outcome="error",
            detail="invalid_request",
        )
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        _record_failed_request(
            request, request_id, None, payload.question, started_at,
            (time.perf_counter() - total_started) * 1000,
            "Research service is temporarily unavailable.",
        )
        _record_audit_safely(
            request, user.user_id, "research", request_id, outcome="error",
            detail="service_unavailable",
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
    response.errors = _safe_client_errors(errors)
    _record_result_metrics(request, result, request_id, None, started_at, total_started)
    _record_audit_safely(request, user.user_id, "research", request_id)
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
    user: AuthenticatedUser = Depends(require_user),
) -> SessionResearchResponse:
    """Research a question and persist the result in an owned session."""
    manager = _session_manager(request)
    try:
        manager.get_session(session_id, owner_id=user.user_id)
    except SessionNotFoundError as error:
        _record_audit_safely(
            request, user.user_id, "session.research", session_id,
            outcome="not_found",
        )
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
        result = router.research_question(payload.question, owner_id=user.user_id)
    except ValueError as error:
        _record_failed_request(
            request, request_id, session_id, payload.question, started_at,
            (time.perf_counter() - total_started) * 1000, str(error),
        )
        _record_audit_safely(
            request, user.user_id, "session.research", session_id,
            outcome="error", detail="invalid_request",
        )
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        _record_failed_request(
            request, request_id, session_id, payload.question, started_at,
            (time.perf_counter() - total_started) * 1000,
            "Research service is temporarily unavailable.",
        )
        _record_audit_safely(
            request, user.user_id, "session.research", session_id,
            outcome="error", detail="service_unavailable",
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
    response.errors = _safe_client_errors(errors)
    try:
        stored_result = manager.add_result(
            session_id=session_id,
            question=response.question,
            answer=response.answer,
            sources=[source.model_dump() for source in response.sources],
            used_local_documents=response.used_local_documents,
            used_web_search=response.used_web_search,
            errors=response.errors,
            owner_id=user.user_id,
        )
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail="Session not found.") from error
    except Exception as error:
        _record_failed_request(
            request, request_id, session_id, payload.question, started_at,
            (time.perf_counter() - total_started) * 1000,
            "Research result could not be saved.",
        )
        _record_audit_safely(
            request, user.user_id, "session.research", session_id,
            outcome="error", detail="persist_failed",
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
    _record_audit_safely(request, user.user_id, "session.research", session_id)
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
    user: AuthenticatedUser = Depends(require_user),
) -> UploadResponse:
    """Validate and process one uploaded TXT, PDF, or DOCX document."""
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if not filename or extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only .txt, .pdf, and .docx files are supported.",
        )

    owner_directory = _owner_directory(user.user_id)
    owner_directory.mkdir(parents=True, exist_ok=True)
    destination = owner_directory / filename
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded document is empty.")

    indexing_result: dict[str, object] | None = None
    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory) / filename
            temporary_path.write_bytes(content)
            indexing_result = process_document(
                temporary_path,
                request.app.state.vector_store,
                owner_id=user.user_id,
            )
        destination.write_bytes(content)
    except Exception as error:
        if indexing_result is not None and str(indexing_result.get("status")) == "indexed":
            try:
                request.app.state.vector_store.delete_document(
                    str(indexing_result["document_id"]),
                    owner_id=user.user_id,
                )
            except Exception:
                logger.exception("Rollback failed for document %s", indexing_result)
        _record_audit_safely(
            request, user.user_id, "document.upload", filename,
            outcome="error", detail="processing_failed",
        )
        raise HTTPException(
            status_code=400,
            detail="The uploaded document could not be processed.",
        ) from error

    document_id = str(indexing_result["document_id"])
    status_value = str(indexing_result["status"])
    _record_audit_safely(
        request, user.user_id, "document.upload", document_id,
        outcome=status_value,
    )
    return UploadResponse(
        filename=filename,
        document_id=document_id,
        chunks_created=int(indexing_result["chunks_created"]),
        status=status_value,
    )


@app.delete("/documents/{document_id}")
def delete_document(
    request: Request,
    document_id: str,
    user: AuthenticatedUser = Depends(require_user),
) -> dict[str, object]:
    """Delete an owned document, its chunks, and its stored file."""
    vector_store: PersistentVectorStore = request.app.state.vector_store
    document = vector_store.get_document(document_id, owner_id=user.user_id)
    if document is None:
        _record_audit_safely(
            request, user.user_id, "document.delete", document_id,
            outcome="not_found",
        )
        raise HTTPException(status_code=404, detail="Document not found.")

    stored_path = _owner_directory(user.user_id) / str(document["filename"])
    try:
        stored_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Stored file could not be removed for %s", document_id)
        _record_audit_safely(
            request, user.user_id, "document.delete", document_id,
            outcome="error", detail="file_removal_failed",
        )
        raise HTTPException(
            status_code=503,
            detail="Document could not be deleted.",
        )

    deleted = vector_store.delete_document(document_id, owner_id=user.user_id)
    if not deleted:
        _record_audit_safely(
            request, user.user_id, "document.delete", document_id,
            outcome="not_found",
        )
        raise HTTPException(status_code=404, detail="Document not found.")

    _record_audit_safely(request, user.user_id, "document.delete", document_id)
    return {"document_id": document_id, "status": "deleted"}
