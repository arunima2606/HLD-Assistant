import pytest

from app import checks, compare, graph, pipeline
from app.db import Document, Entity, Project, session_scope
from app.extract import load_entities


@pytest.fixture
def docs(sample_pdfs):
    with session_scope() as s:
        p = Project(name="bcm")
        s.add(p)
        s.commit()
        ids = []
        for ver, key in (("1.0", "v1.0"), ("1.1", "v1.1")):
            d = pipeline.register_document(s, p.id, sample_pdfs[key].name, ver, sample_pdfs[key])
            ids.append(d.id)
        pid = p.id
    for i in ids:
        pipeline.process_document(i)
    return pid, ids


def test_planted_inconsistencies_detected_and_nothing_else(docs):
    pid, (v10, _) = docs
    with session_scope() as s:
        found = checks.run_checks(s, pid, v10)
    got = {(f.rule_id, f.entities[0]) for f in found}
    assert got == {("R001", "DiagnosticHandler.DoorLockDiag"), ("R002", "DiagSessionState"),
                   ("R003", "LightingController -> AmbientSensorDriver")}, got
    assert all(f.citations and f.citations[0]["page"] for f in found)  # every finding is traceable
    assert found[0].severity == "high"


def test_checks_cover_both_versions_and_skip_rejected(docs):
    pid, (v10, v11) = docs
    with session_scope() as s:
        assert {f.doc_id for f in checks.run_checks(s, pid)} == {v10, v11}
        sig = s.query(Entity).filter_by(document_id=v10, type="signal", name="DiagSessionState").one()
        sig.review_status = "rejected"
        s.commit()
        assert "R002" not in {f.rule_id for f in checks.run_checks(s, pid, v10)}


def test_rules_r004_r005_r006_r007():
    def E(t, name, key, **a):
        return Entity(type=t, name=name, key=key, attrs=a, sources=[{"page": 1, "section": "x"}])
    ents = [E("component", "DoorLockManager", "doorlockmanager"), E("component", "DoorLockManger", "doorlockmanger"),
            E("component", "Lonely", "lonely"),
            E("port", "Ghost.P", "ghost.p", component="Ghost", port="P", direction="provided", interface="I"),
            E("signal", "S", "s", sender="Nobody", receiver="DoorLockManager")]
    rules = {f.rule_id for f in checks.check_entities(ents, 1)}
    assert {"R004", "R005", "R006", "R007"} <= rules


def test_graph_structure_and_html(docs):
    _, (v10, _) = docs
    with session_scope() as s:
        g = graph.build_graph(load_entities(s, v10))
    kinds = {d["kind"] for _, d in g.nodes(data=True)}
    assert kinds == {"component", "interface", "undefined"}  # AmbientSensorDriver is flagged as undefined
    assert g.has_edge("c:doorlockmanager", "c:powermanager")
    data = graph.graph_json(g)
    assert len(data["nodes"]) == g.number_of_nodes() and len(data["edges"]) == g.number_of_edges()
    html = graph.graph_html(g)
    assert "<html" in html.lower() and "DoorLockManager" in html and "unpkg.com" not in html


def test_compare_v10_v11(docs):
    _, (v10, v11) = docs
    with session_scope() as s:
        r = compare.compare_documents(s, v10, v11)
    names = lambda items: {(i["type"], i["name"]) for i in items}  # noqa: E731
    assert names(r["added"]) == {("component", "RainSensorHandler"), ("interface", "IRainStatus"),
                                 ("port", "RainSensorHandler.RainStatusOut"), ("signal", "RainIntensity"),
                                 ("dependency", "WindowController -> RainSensorHandler")}
    assert names(r["removed"]) == {("port", "WindowController.PowerStatePort")}
    assert [(x["from"], x["to"]) for x in r["renamed"]] == [("VehicleSpeed", "VehSpeed")]
    assert r["changed"] == []
    assert r["added"][0]["citation"]["version"] == "1.1" and r["removed"][0]["citation"]["version"] == "1.0"


def test_compare_detects_attribute_change(docs):
    _, (v10, v11) = docs
    with session_scope() as s:
        e = s.query(Entity).filter_by(document_id=v11, type="signal", name="LightRequest").one()
        e.attrs = {**e.attrs, "receiver": "PowerManager"}
        s.commit()
        r = compare.compare_documents(s, v10, v11)
    ch = next(c for c in r["changed"] if c["name"] == "LightRequest")
    assert ch["changes"]["receiver"] == {"from": "LightingController", "to": "PowerManager"}


def test_compare_errors(docs):
    with session_scope() as s:
        with pytest.raises(LookupError):
            compare.compare_documents(s, 1, 9999)


def test_graph_html_escapes_malicious_names():
    """Names come from uploaded PDFs; they must not be able to break out of pyvis's <script> block."""
    evil = "</script><script>alert(1)</script>"
    ents = [Entity(type="component", name=evil, key="evil", attrs={}, sources=[]),
            Entity(type="dependency", name="d", key="d", attrs={"from": evil, "to": "B", "dep_type": "<img src=x onerror=alert(2)>"},
                   sources=[])]
    html = graph.graph_html(graph.build_graph(ents))
    assert "<script>alert(1)" not in html and "</script><script>alert" not in html
    assert "<img src=x onerror" not in html
    # neutralised payload is present (pyvis JSON-encodes '&' as \\u0026, so accept either spelling)
    assert "&lt;/script" in html or "\\u0026lt;/script" in html
