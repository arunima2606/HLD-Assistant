"""Audit log helper: who asked/changed what, when, with which sources."""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db import AuditLog


def log_action(session: Session, *, action: str, user: str = "anonymous", role: str = "viewer",
               project_id: int | None = None, query: str = "", sources: list[Any] | None = None,
               details: dict[str, Any] | None = None) -> None:
    session.add(AuditLog(user=user, role=role, action=action, project_id=project_id,
                         query=query, sources=sources or [], details=details or {}))
    session.commit()
