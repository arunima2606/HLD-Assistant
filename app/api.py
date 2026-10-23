"""FastAPI service. Role handling (X-Role header) is a PILOT-LEVEL stand-in for real authentication/RBAC."""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import re
import uuid
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Query, Response, UploadFile
from sqlalchemy.orm import Session

from app import checks, compare, graph, index, llm, pipeline, rag
from app.audit import log_action
from app.config import get_settings, setup_logging
from app.db import AuditLog, Document, Entity, Project, get_session
from app.extract import load_entities
from app.schemas import (AskRequest, AskResponse, CompareRequest, DocumentOut, EntityOut, EntityPatch, Finding,
                         ProjectCreate, ProjectOut)
from app.textutil import norm_key

setup_logging()
app = FastAPI(title="AUTOSAR HLD Analysis Assistant", version="1.0.0",
              description="Local RAG assistant for AUTOSAR High-Level Design documents. "
                          "AI output is advisory: all extracted entities start as 'pending' until a human approves them.")

Role = Literal["admin", "viewer"]


class Caller:
    def __init__(self, role: str, user: str):
        self.role, self.user = role, user


def caller(x_role: str = Header("viewer"), x_user: str = Header("anonymous")) -> Caller:
    if x_role not in ("admin", "viewer"):
        raise HTTPException(400, "X-Role must be 'admin' or 'viewer'")
    return Caller(x_role, x_user[:100])


def admin(c: Caller = Depends(caller)) -> Caller:
    if c.role != "admin":
        raise HTTPException(403, "This action requires the 'admin' role (X-Role: admin)")
    return c


def _project(session: Session, project_id: int) -> Project:
    p = session.get(Project, project_id)
    if not p:
        raise HTTPException(404, f"Project {project_id} not found")
    return p


def _doc(session: Session, project_id: int, doc_id: int) -> Document:
    d = session.get(Document, doc_id)
    if not d or d.project_id != project_id:
        raise HTTPException(404, f"Document {doc_id} not found in project {project_id}")
    return d


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {"status": "ok", "mock_llm": s.mock_llm, "llm_model": s.llm_model, "embedding_backend": s.embedding_backend}


# ---------------- projects & documents ----------------
@app.post("/projects", response_model=ProjectOut, status_code=201)
def create_project(body: ProjectCreate, c: Caller = Depends(admin), session: Session = Depends(get_session)):
    if session.query(Project).filter_by(name=body.name).first():
        raise HTTPException(409, f"Project '{body.name}' already exists")
    p = Project(name=body.name, description=body.description)
    session.add(p)
    session.commit()
    log_action(session, action="create_project", user=c.user, role=c.role, project_id=p.id, details={"name": p.name})
    return p


@app.get("/projects", response_model=list[ProjectOut])
def list_projects(session: Session = Depends(get_session)):
    return session.query(Project).order_by(Project.name).all()


@app.get("/projects/{project_id}/documents", response_model=list[DocumentOut])
def list_documents(project_id: int, session: Session = Depends(get_session)):
    _project(session, project_id)
    return session.query(Document).filter_by(project_id=project_id).order_by(Document.id).all()


@app.post("/projects/{project_id}/upload", response_model=DocumentOut, status_code=202)
def upload(project_id: int, background: BackgroundTasks, file: UploadFile = File(...), version: str = Form("1.0"),
           c: Caller = Depends(admin), session: Session = Depends(get_session)):
    """Store the PDF, then ingest -> index -> extract in the background. Poll the status endpoint."""
    _project(session, project_id)
    s = get_settings()
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(file.filename or "upload.pdf").name)
    if not name.lower().endswith(".pdf"):
        raise HTTPException(415, "Only PDF files are supported")
    data = file.file.read(s.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {s.max_upload_mb} MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "File does not look like a PDF")
    folder = s.upload_dir / str(project_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex[:8]}_{name}"
    path.write_bytes(data)
    doc = pipeline.register_document(session, project_id, name, version.strip() or "1.0", path)
    log_action(session, action="upload", user=c.user, role=c.role, project_id=project_id,
               details={"doc_id": doc.id, "filename": name, "version": doc.version})
    background.add_task(pipeline.process_document, doc.id)
    return doc


@app.get("/projects/{project_id}/documents/{doc_id}/status", response_model=DocumentOut)
def document_status(project_id: int, doc_id: int, session: Session = Depends(get_session)):
    return _doc(session, project_id, doc_id)


# ---------------- Q&A ----------------
@app.post("/projects/{project_id}/ask", response_model=AskResponse)
def ask(project_id: int, body: AskRequest, c: Caller = Depends(caller), session: Session = Depends(get_session)):
    _project(session, project_id)
    if not session.query(Document).filter_by(project_id=project_id, status="ready").count():
        raise HTTPException(409, "No indexed documents yet - upload a PDF and wait for status 'ready'")
    if body.doc_id is not None:
        _doc(session, project_id, body.doc_id)
    try:
        return rag.ask(project_id, body.question, k=body.k, doc_id=body.doc_id, session=session, user=c.user, role=c.role)
    except llm.LLMError as exc:
        raise HTTPException(503, str(exc))
    except index.IndexError_ as exc:
        raise HTTPException(500, str(exc))


# ---------------- entities ----------------
@app.get("/projects/{project_id}/entities", response_model=list[EntityOut])
def list_entities(project_id: int, type: str | None = None, doc: int | None = None,
                  status: Literal["pending", "approved", "rejected"] | None = None, session: Session = Depends(get_session)):
    _project(session, project_id)
    q = session.query(Entity).filter_by(project_id=project_id)
    if type:
        q = q.filter_by(type=type)
    if doc is not None:
        q = q.filter_by(document_id=doc)
    if status:
        q = q.filter_by(review_status=status)
    return q.order_by(Entity.document_id, Entity.type, Entity.name).all()


@app.patch("/entities/{entity_id}", response_model=EntityOut)
def patch_entity(entity_id: int, body: EntityPatch, c: Caller = Depends(admin), session: Session = Depends(get_session)):
    """Edit fields and/or approve/reject. Human review is the control point for all AI-extracted data."""
    e = session.get(Entity, entity_id)
    if not e:
        raise HTTPException(404, f"Entity {entity_id} not found")
    before = {"name": e.name, "attrs": dict(e.attrs), "review_status": e.review_status}
    if body.name is not None:
        if e.type in ("port", "dependency"):
            raise HTTPException(422, f"Edit '{e.type}' via attrs; its name is derived from them")
        if not body.name.strip():
            raise HTTPException(422, "name must not be empty")
        e.name, e.key = body.name.strip(), norm_key(body.name)
    if body.attrs is not None:
        attrs = {**e.attrs, **{k: ("" if v is None else str(v).strip()) for k, v in body.attrs.items()}}
        if e.type == "port":
            if attrs.get("direction") not in ("provided", "required"):
                raise HTTPException(422, "direction must be 'provided' or 'required'")
            e.name, e.key = f"{attrs['component']}.{attrs['port']}", f"{norm_key(attrs['component'])}.{norm_key(attrs['port'])}"
        elif e.type == "dependency":
            e.name, e.key = f"{attrs['from']} -> {attrs['to']}", f"{norm_key(attrs['from'])}->{norm_key(attrs['to'])}"
        e.attrs = attrs
    if body.review_status is not None:
        e.review_status = body.review_status
    with session.no_autoflush:  # check before flushing, otherwise the unique constraint fires first (500)
        clash = session.query(Entity).filter(Entity.document_id == e.document_id, Entity.type == e.type,
                                             Entity.key == e.key, Entity.id != e.id).first()
    if clash:
        session.rollback()
        raise HTTPException(409, f"Another {e.type} with the same name already exists in this document")
    e.edited = e.edited or body.name is not None or body.attrs is not None
    session.commit()
    log_action(session, action="edit_entity", user=c.user, role=c.role, project_id=e.project_id,
               details={"entity_id": e.id, "before": before, "after": {"name": e.name, "attrs": e.attrs,
                                                                       "review_status": e.review_status}})
    return e


# ---------------- analysis ----------------
def _doc_or_latest(session: Session, project_id: int, doc: int | None) -> Document:
    if doc is not None:
        return _doc(session, project_id, doc)
    d = session.query(Document).filter_by(project_id=project_id, status="ready").order_by(Document.id.desc()).first()
    if not d:
        raise HTTPException(409, "No processed documents in this project yet")
    return d


@app.get("/projects/{project_id}/graph")
def get_graph(project_id: int, doc: int | None = None, session: Session = Depends(get_session)):
    _project(session, project_id)
    d = _doc_or_latest(session, project_id, doc)
    g = graph.build_graph(load_entities(session, d.id))
    return {"doc_id": d.id, "version": d.version, **graph.graph_json(g), "html": graph.graph_html(g)}


@app.get("/projects/{project_id}/checks", response_model=list[Finding])
def get_checks(project_id: int, doc: int | None = None, session: Session = Depends(get_session)):
    _project(session, project_id)
    if doc is not None:
        _doc(session, project_id, doc)
    return checks.run_checks(session, project_id, doc)


@app.post("/projects/{project_id}/compare")
def compare_versions(project_id: int, body: CompareRequest, c: Caller = Depends(caller),
                     session: Session = Depends(get_session)):
    _project(session, project_id)
    _doc(session, project_id, body.doc_a)
    _doc(session, project_id, body.doc_b)
    result = compare.compare_documents(session, body.doc_a, body.doc_b)
    log_action(session, action="compare", user=c.user, role=c.role, project_id=project_id, details=result["summary"])
    return result


# ---------------- export & audit ----------------
@app.get("/projects/{project_id}/export")
def export(project_id: int, format: Literal["csv", "json"] = "json", doc: int | None = None,
           c: Caller = Depends(caller), session: Session = Depends(get_session)):
    p = _project(session, project_id)
    docs = {d.id: d for d in session.query(Document).filter_by(project_id=project_id).all()}
    q = session.query(Entity).filter_by(project_id=project_id)
    if doc is not None:
        _doc(session, project_id, doc)
        q = q.filter_by(document_id=doc)
    ents = q.order_by(Entity.document_id, Entity.type, Entity.name).all()
    log_action(session, action="export", user=c.user, role=c.role, project_id=project_id,
               details={"format": format, "entities": len(ents)})
    row = lambda e: {"type": e.type, "name": e.name, **{f"attr_{k}": v for k, v in e.attrs.items()},  # noqa: E731
                     "review_status": e.review_status, "edited": e.edited, "doc": docs[e.document_id].filename,
                     "version": docs[e.document_id].version, "pages": sorted({s["page"] for s in e.sources}),
                     "sections": sorted({s["section"] for s in e.sources})}
    if format == "csv":
        rows = [row(e) for e in ents]
        cols = ["type", "name", "attr_component", "attr_port", "attr_direction", "attr_interface", "attr_sender",
                "attr_receiver", "attr_from", "attr_to", "attr_dep_type", "attr_description", "review_status",
                "edited", "doc", "version", "pages", "sections"]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r["pages"], r["sections"] = ";".join(map(str, r["pages"])), ";".join(r["sections"])
            w.writerow(r)
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{p.name}_entities.csv"'})
    grouped: dict[str, list] = {}
    for e in ents:
        grouped.setdefault(e.type, []).append(row(e))
    payload = {"project": p.name, "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "note": "AI-extracted content; only 'approved' entries have been reviewed by a human.",
               "documents": [{"id": d.id, "filename": d.filename, "version": d.version} for d in docs.values()],
               "entities": grouped, "findings": [f.model_dump() for f in checks.run_checks(session, project_id, doc)]}
    return Response(json.dumps(payload, indent=2), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{p.name}_export.json"'})


@app.get("/projects/{project_id}/audit")
def audit_log(project_id: int, limit: int = Query(200, ge=1, le=2000), c: Caller = Depends(admin),
              session: Session = Depends(get_session)):
    _project(session, project_id)
    rows = session.query(AuditLog).filter_by(project_id=project_id).order_by(AuditLog.id.desc()).limit(limit).all()
    return [{"id": r.id, "ts": r.ts.isoformat(), "user": r.user, "role": r.role, "action": r.action,
             "query": r.query, "sources": r.sources, "details": r.details} for r in rows]
