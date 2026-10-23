"""Pydantic models: API contracts + the strict shape we require from LLM extraction."""
from __future__ import annotations

import datetime as dt
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from app.textutil import clean


def _s(v: Any) -> str:
    return clean(v)


def _direction(v: Any) -> str:
    d = clean(v).lower().replace("_", "").replace("-", "").replace(" ", "")
    if d.startswith("prov") or d in {"p", "pport", "out"}:
        return "provided"
    if d.startswith("req") or d in {"r", "rport", "in"}:
        return "required"
    return ""


Str = Annotated[str, BeforeValidator(_s)]
Direction = Annotated[str, BeforeValidator(_direction)]


# ---------- LLM extraction contract (lenient on input, strict on output) ----------
class ExtComponent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: Str = ""
    description: Str = ""


class ExtInterface(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: Str = ""
    description: Str = ""


class ExtPort(BaseModel):
    model_config = ConfigDict(extra="ignore")
    component: Str = ""
    name: Str = ""
    direction: Direction = ""
    interface: Str = ""


class ExtSignal(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: Str = ""
    sender: Str = ""
    receiver: Str = ""


class ExtDependency(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    from_: Str = Field("", alias="from")
    to: Str = ""
    type: Str = ""


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    components: list[ExtComponent] = []
    interfaces: list[ExtInterface] = []
    ports: list[ExtPort] = []
    signals: list[ExtSignal] = []
    dependencies: list[ExtDependency] = []

    def is_empty(self) -> bool:
        return not (self.components or self.interfaces or self.ports or self.signals or self.dependencies)


# ---------- API models ----------
EntityType = Literal["component", "interface", "port", "signal", "dependency"]
ReviewStatus = Literal["pending", "approved", "rejected"]


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    description: str
    created_at: dt.datetime


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    project_id: int
    filename: str
    version: str
    status: str
    message: str
    n_pages: int
    n_chunks: int
    n_entities: int
    created_at: dt.datetime


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    k: int = Field(6, ge=1, le=20)
    doc_id: int | None = None


class Source(BaseModel):
    doc: str
    doc_id: int
    version: str
    section: str
    page_start: int
    page_end: int
    type: str
    similarity: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    confidence: float
    confidence_label: Literal["high", "medium", "low", "none"]
    not_found: bool


class EntityOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    project_id: int
    document_id: int
    type: str
    name: str
    attrs: dict[str, Any]
    sources: list[dict[str, Any]]
    review_status: str
    edited: bool


class EntityPatch(BaseModel):
    name: str | None = None
    attrs: dict[str, Any] | None = None
    review_status: ReviewStatus | None = None


class Finding(BaseModel):
    rule_id: str
    severity: Literal["high", "medium", "low"]
    message: str
    doc_id: int
    version: str = ""
    entities: list[str] = []
    citations: list[dict[str, Any]] = []


class CompareRequest(BaseModel):
    doc_a: int
    doc_b: int
