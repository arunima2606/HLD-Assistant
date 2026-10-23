"""Entity-level diff between two document versions (not raw text)."""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db import Document, Entity
from app.extract import load_entities
from app.textutil import norm_key

IGNORED_ATTRS = {"description"}  # wording tweaks are noise; structure is what matters


def _item(e: Entity, doc: Document) -> dict[str, Any]:
    src = e.sources[0] if e.sources else {}
    return {"type": e.type, "name": e.name, "attrs": e.attrs,
            "citation": {"doc": doc.filename, "version": doc.version, "page": src.get("page"), "section": src.get("section")}}


def _same(a: dict, b: dict) -> list[str]:
    return [k for k in sorted(set(a) | set(b) - IGNORED_ATTRS)
            if k not in IGNORED_ATTRS and norm_key(str(a.get(k, ""))) != norm_key(str(b.get(k, "")))]


def compare_entities(ents_a: list[Entity], doc_a: Document, ents_b: list[Entity], doc_b: Document) -> dict[str, Any]:
    A = {(e.type, e.key): e for e in ents_a}
    B = {(e.type, e.key): e for e in ents_b}
    added = [_item(B[k], doc_b) for k in sorted(set(B) - set(A))]
    removed = [_item(A[k], doc_a) for k in sorted(set(A) - set(B))]
    changed = []
    for k in sorted(set(A) & set(B)):
        diff = _same(A[k].attrs, B[k].attrs)
        if diff:
            changed.append({"type": k[0], "name": B[k].name,
                            "changes": {f: {"from": A[k].attrs.get(f, ""), "to": B[k].attrs.get(f, "")} for f in diff},
                            "citation_a": _item(A[k], doc_a)["citation"], "citation_b": _item(B[k], doc_b)["citation"]})

    # A removed + an added signal with identical endpoints is almost certainly a rename.
    renamed = []
    for r in list(removed):
        if r["type"] != "signal" or not (r["attrs"].get("sender") and r["attrs"].get("receiver")):
            continue
        for a in list(added):
            if a["type"] == "signal" and (a["attrs"].get("sender"), a["attrs"].get("receiver")) == \
                    (r["attrs"]["sender"], r["attrs"]["receiver"]):
                renamed.append({"type": "signal", "from": r["name"], "to": a["name"],
                                "citation_a": r["citation"], "citation_b": a["citation"]})
                removed.remove(r)
                added.remove(a)
                break
    return {"doc_a": {"id": doc_a.id, "filename": doc_a.filename, "version": doc_a.version},
            "doc_b": {"id": doc_b.id, "filename": doc_b.filename, "version": doc_b.version},
            "added": added, "removed": removed, "changed": changed, "renamed": renamed,
            "summary": {"added": len(added), "removed": len(removed), "changed": len(changed), "renamed": len(renamed)}}


def compare_documents(session: Session, doc_a_id: int, doc_b_id: int) -> dict[str, Any]:
    a, b = session.get(Document, doc_a_id), session.get(Document, doc_b_id)
    if not a or not b:
        raise LookupError("Both documents must exist")
    if a.project_id != b.project_id:
        raise ValueError("Documents belong to different projects")
    return compare_entities(load_entities(session, a.id), a, load_entities(session, b.id), b)
