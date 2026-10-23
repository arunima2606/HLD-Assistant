import json

import pytest

from app import extract, llm, pipeline
from app.db import Document, Entity, Project, session_scope
from app.schemas import ExtractionResult
from scripts.evaluate_extraction import evaluate


def test_mock_extraction_matches_ground_truth(sample_pdfs, tmp_path):
    gt = tmp_path / "gt.json"
    from scripts.make_sample_hld import SPEC_V10, ground_truth
    gt.write_text(json.dumps(ground_truth(SPEC_V10)))
    rep = evaluate(sample_pdfs["v1.0"], gt)
    for typ, m in rep.items():
        assert (m["precision"], m["recall"], m["attr_accuracy"]) == (1.0, 1.0, 1.0), (typ, m)


def _process(sample_pdf, version="1.0"):
    with session_scope() as s:
        p = s.query(Project).first() or Project(name="bcm")
        s.add(p)
        s.commit()
        d = pipeline.register_document(s, p.id, sample_pdf.name, version, sample_pdf)
        pid, did = p.id, d.id
    pipeline.process_document(did)
    return pid, did


def test_pipeline_persists_entities_with_citations(sample_pdfs):
    _, did = _process(sample_pdfs["v1.0"])
    with session_scope() as s:
        doc = s.get(Document, did)
        assert doc.status == "ready", doc.message
        assert doc.n_pages == 7 and doc.n_entities == 6 + 6 + 10 + 6 + 6
        port = s.query(Entity).filter_by(document_id=did, type="port", key="doorlockmanager.doorlockcmd").one()
        assert port.review_status == "pending" and port.sources[0]["page"] == 4
        assert port.attrs["direction"] == "provided"


def test_reprocessing_is_idempotent(sample_pdfs):
    pid, did = _process(sample_pdfs["v1.0"])
    pipeline.process_document(did)
    with session_scope() as s:
        assert s.query(Entity).filter_by(document_id=did).count() == 34


def test_bad_pdf_marks_document_failed(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 garbage")
    _, did = _process(bad)
    with session_scope() as s:
        d = s.get(Document, did)
        assert d.status == "failed" and d.message


def test_merge_dedupes_names_and_accumulates_pages():
    acc: dict = {}
    r1 = ExtractionResult.model_validate({"components": [{"name": "Door_Lock Manager", "description": ""}]})
    r2 = ExtractionResult.model_validate({"components": [{"name": "DoorLockManager", "description": "locks doors"}]})
    extract.merge_into(acc, r1, 2, "2.1", "llm")
    extract.merge_into(acc, r2, 4, "3.2", "llm")
    (a,) = acc.values()
    assert a.attrs["description"] == "locks doors" and [s["page"] for s in a.sources] == [2, 4]


def test_json_retry_then_skip(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "generate", lambda *a, **k: calls.append(1) or "sorry, not json")
    assert llm.generate_json("p", ExtractionResult, retries=1) is None and len(calls) == 2
    outs = iter(["```json\n{\"components\":[{\"name\":\"A\"}]}\n```"])
    monkeypatch.setattr(llm, "generate", lambda *a, **k: next(outs))
    assert llm.generate_json("p", ExtractionResult).components[0].name == "A"


def test_llm_output_is_grounded_and_lenient(monkeypatch):
    raw = ('{"components":[{"name":"RealComp","description":null},{"name":"Hallucinated"}],'
           '"ports":[{"component":"RealComp","name":"P1","direction":"R-Port","interface":null}]}')
    monkeypatch.setattr(llm, "generate", lambda *a, **k: raw)
    res = extract.extract_with_llm("The RealComp has port P1 for power.", "x")
    assert [c.name for c in res.components] == ["RealComp"] and res.ports[0].direction == "required"


def test_llm_down_keeps_deterministic_results(monkeypatch, sample_pdfs):
    def boom(*a, **k):
        raise llm.LLMError("down")
    monkeypatch.setattr(llm, "generate", boom)
    _, did = _process(sample_pdfs["v1.0"])
    with session_scope() as s:
        d = s.get(Document, did)
        assert d.status == "ready" and d.n_entities == 34 and "LLM errors" in d.message


def test_concurrent_processing_regression(sample_pdfs):
    """Real servers run background tasks in parallel threads; both documents must still succeed."""
    import threading
    with session_scope() as s:
        p = Project(name="race")
        s.add(p)
        s.commit()
        ids = [pipeline.register_document(s, p.id, sample_pdfs[k].name, v, sample_pdfs[k]).id
               for k, v in (("v1.0", "1.0"), ("v1.1", "1.1"))]
    threads = [threading.Thread(target=pipeline.process_document, args=(i,)) for i in ids]
    [t.start() for t in threads]
    [t.join() for t in threads]
    with session_scope() as s:
        assert [s.get(Document, i).status for i in ids] == ["ready", "ready"], [s.get(Document, i).message for i in ids]
