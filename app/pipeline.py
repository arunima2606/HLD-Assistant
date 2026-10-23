"""Background processing: ingest -> index -> extract, with status updates for polling."""
from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

from app import extract, index
from app.db import ChunkRow, Document, session_scope
from app.ingest import IngestError, ingest_pdf

log = logging.getLogger(__name__)
# One document at a time: the embedding model, Chroma and SQLite writers are not built for parallel ingestion.
_PROCESS_LOCK = threading.Lock()


def _status(doc_id: int, status: str, message: str = "", **fields) -> None:
    with session_scope() as s:
        d = s.get(Document, doc_id)
        d.status, d.message = status, message
        for k, v in fields.items():
            setattr(d, k, v)


def register_document(session, project_id: int, filename: str, version: str, path: Path) -> Document:
    doc = Document(project_id=project_id, filename=filename, version=version, path=str(path), status="queued",
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    session.add(doc)
    session.commit()
    return doc


def process_document(doc_id: int) -> None:
    """Never raises: failures are recorded on the document row. Concurrent calls are queued."""
    with _PROCESS_LOCK:
        _process(doc_id)


def _process(doc_id: int) -> None:
    try:
        with session_scope() as s:
            d = s.get(Document, doc_id)
            project_id, path, version, filename = d.project_id, d.path, d.version, d.filename
        _status(doc_id, "processing", "parsing PDF")
        parsed, chunks = ingest_pdf(path, version)
        for c in chunks:
            c.doc = filename  # cite the original upload name, not the stored file name

        _status(doc_id, "processing", f"indexing {len(chunks)} chunks", n_pages=parsed.n_pages, n_chunks=len(chunks))
        index.delete_document(project_id, doc_id)
        ids = index.index_chunks(project_id, doc_id, chunks)
        with session_scope() as s:
            s.query(ChunkRow).filter_by(document_id=doc_id).delete()
            s.add_all(ChunkRow(document_id=doc_id, chunk_index=c.index, chroma_id=i, section=c.section,
                               page_start=c.page_start, page_end=c.page_end, type=c.type, text=c.text)
                      for c, i in zip(chunks, ids))

        _status(doc_id, "processing", "extracting entities")
        with session_scope() as s:
            summary = extract.extract_document(s, project_id, doc_id)
        note = f"{summary['total']} entities"
        if summary["llm_failed"] or summary["llm_errors"]:
            note += f" ({summary['llm_failed']} chunks skipped: invalid LLM output, {summary['llm_errors']} LLM errors)"
        if parsed.ocr_pages:
            note += f"; OCR used on pages {parsed.ocr_pages}"
        _status(doc_id, "ready", note, n_entities=summary["total"])
    except IngestError as exc:
        log.error("Ingest failed for doc %s: %s", doc_id, exc)
        _status(doc_id, "failed", str(exc))
    except Exception as exc:  # noqa: BLE001 - background task must not crash the server
        log.exception("Processing failed for doc %s", doc_id)
        _status(doc_id, "failed", f"{type(exc).__name__}: {exc}")
