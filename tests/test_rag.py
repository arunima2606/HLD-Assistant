import pytest

from app import index, rag
from app.db import AuditLog, Project, session_scope
from app.ingest import ingest_pdf


@pytest.fixture
def project(sample_pdfs):
    with session_scope() as s:
        p = Project(name="bcm")
        s.add(p)
        s.commit()
        pid = p.id
    _, chunks = ingest_pdf(sample_pdfs["v1.0"], "1.0")
    index.index_chunks(pid, 1, chunks)
    return pid


QUESTIONS = [
    ("Which component is responsible for locking the doors?", "2."),
    ("What signals does the ComSignalRouter exchange?", "4.1"),
    ("What does the DiagnosticHandler depend on?", "4.2"),
    ("Which ports does the PowerManager provide?", "3.2"),
    ("What is the purpose of this document?", "1.1"),
]


@pytest.mark.parametrize("question,section_prefix", QUESTIONS)
def test_retrieval_finds_right_section(project, question, section_prefix):
    hits = index.query(project, question, k=3)
    assert any(h["meta"]["section"].startswith(section_prefix) for h in hits), [h["meta"]["section"] for h in hits]


def test_ask_returns_sources_and_audits(project):
    with session_scope() as s:
        r = rag.ask(project, "Which ports does the PowerManager provide?", session=s, user="arunima", role="admin")
        assert not r.not_found and r.sources and r.sources[0].page_start >= 1
        assert r.confidence_label in {"high", "medium", "low"} and 0 < r.confidence <= 1
        assert "[sample_hld_v1.0.pdf" in r.answer  # mock answer carries the citation label
        log = s.query(AuditLog).filter_by(action="ask").one()
        assert log.user == "arunima" and log.sources


def test_unrelated_question_is_not_found(project):
    r = rag.ask(project, "zzqx blorptastic quantum banana?")
    assert r.not_found and r.sources == [] and r.confidence_label == "none"


def test_empty_project_is_graceful():
    r = rag.ask(999, "anything at all?")
    assert r.not_found


def test_doc_filter(project):
    assert index.query(project, "ports", k=3, doc_id=42) == []
    assert index.query(project, "ports", k=3, doc_id=1)
