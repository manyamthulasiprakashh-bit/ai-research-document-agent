import hashlib
from pathlib import Path

from app.ingestion.document_loader import load_document
from app.processing.text_processor import chunk_text, clean_text
from app.retrieval.embeddings import create_embedding
from app.retrieval.vector_store import VectorStore

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def discover_documents(
    directory: str | Path = "data/documents",
) -> list[Path]:
    """Return supported documents in a directory, sorted by filename."""
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"Document directory not found: {directory}")

    return sorted(
        (
            path
            for path in directory_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def process_documents(
    directory: str | Path,
    vector_store: VectorStore,
    chunk_size: int = 30,
    overlap: int = 5,
) -> list[str]:
    """Load, process, embed, and store every supported document."""
    processed_documents: list[str] = []

    for document_path in discover_documents(directory):
        process_document(
            document_path,
            vector_store,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        processed_documents.append(document_path.name)

    return processed_documents


def process_document(
    file_path: str | Path,
    vector_store: VectorStore,
    chunk_size: int = 30,
    overlap: int = 5,
) -> dict[str, object]:
    """Process one document and index it, skipping duplicate content."""
    document_path = Path(file_path)
    content = document_path.read_bytes()
    fingerprint = hashlib.sha256(content).hexdigest()
    document_id = f"doc-{fingerprint}"

    get_existing = getattr(vector_store, "get_document_by_fingerprint", None)
    if get_existing is not None:
        existing = get_existing(fingerprint)
        if existing:
            return {
                "document_id": existing["document_id"],
                "filename": existing["filename"],
                "chunks_created": existing["chunks"],
                "status": "duplicate",
            }

    text = load_document(str(document_path))
    cleaned_text = clean_text(text)
    chunks = chunk_text(
        cleaned_text,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    embedded_chunks = [
        (
            f"{document_path.name}:chunk-{index}",
            chunk,
            create_embedding(chunk),
        )
        for index, chunk in enumerate(chunks, start=1)
    ]

    add_document = getattr(vector_store, "add_document", None)
    if add_document is not None:
        return add_document(
            document_id=document_id,
            filename=document_path.name,
            fingerprint=fingerprint,
            chunks=embedded_chunks,
        )

    for chunk_id, chunk, embedding in embedded_chunks:
        vector_store.add(
            chunk_id=chunk_id,
            text=chunk,
            embedding=embedding,
            source=document_path.name,
        )

    return {
        "document_id": document_id,
        "filename": document_path.name,
        "chunks_created": len(embedded_chunks),
        "status": "indexed",
    }
