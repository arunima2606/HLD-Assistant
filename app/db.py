"""SQLAlchemy models + session helpers (SQLite for the pilot, PostgreSQL-ready)."""
from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from app.config import get_settings


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    filename: Mapped[str] = mapped_column(String(300))
    version: Mapped[str] = mapped_column(String(50), default="1.0")
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued|processing|ready|failed
    message: Mapped[str] = mapped_column(Text, default="")
    path: Mapped[str] = mapped_column(String(600), default="")
    sha256: Mapped[str] = mapped_column(String(64), default="")
    n_pages: Mapped[int] = mapped_column(Integer, default=0)
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    n_entities: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)


class ChunkRow(Base):
    """Chunk text + metadata (the vector lives in Chroma under `chroma_id`)."""

    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    chroma_id: Mapped[str] = mapped_column(String(100))
    section: Mapped[str] = mapped_column(String(300), default="")
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(10), default="text")  # text|table
    text: Mapped[str] = mapped_column(Text)


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("document_id", "type", "key", name="uq_entity"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    type: Mapped[str] = mapped_column(String(20), index=True)  # component|interface|port|signal|dependency
    name: Mapped[str] = mapped_column(String(300))
    key: Mapped[str] = mapped_column(String(400))
    attrs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # [{page, section, origin}]
    review_status: Mapped[str] = mapped_column(String(10), default="pending")  # pending|approved|rejected
    edited: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)
    user: Mapped[str] = mapped_column(String(100), default="anonymous")
    role: Mapped[str] = mapped_column(String(20), default="viewer")
    action: Mapped[str] = mapped_column(String(50))
    project_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    query: Mapped[str] = mapped_column(Text, default="")
    sources: Mapped[list[Any]] = mapped_column(JSON, default=list)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


_engines: dict[str, Engine] = {}


def get_engine(url: str | None = None) -> Engine:
    """Return a cached engine for the configured DB, creating tables on first use."""
    url = url or get_settings().database_url
    if url not in _engines:
        kwargs: dict[str, Any] = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        engine = create_engine(url, **kwargs)
        Base.metadata.create_all(engine)
        _engines[url] = engine
    return _engines[url]


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for scripts and background tasks."""
    session = Session(get_engine(), expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    session = Session(get_engine(), expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()
