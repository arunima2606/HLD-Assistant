import pytest
from fastapi.testclient import TestClient

from app.api import app

ADMIN = {"X-Role": "admin", "X-User": "arunima"}
VIEWER = {"X-Role": "viewer", "X-User": "guest"}


@pytest.fixture
def client():
    return TestClient(app)


def upload(client, pdf, pid, version):
    with open(pdf, "rb") as f:
        r = client.post(f"/projects/{pid}/upload", headers=ADMIN, files={"file": (pdf.name, f, "application/pdf")},
                        data={"version": version})
    assert r.status_code == 202, r.text
    return r.json()["id"]


@pytest.fixture
def loaded(client, sample_pdfs):
    pid = client.post("/projects", headers=ADMIN, json={"name": "BCM"}).json()["id"]
    d1 = upload(client, sample_pdfs["v1.0"], pid, "1.0")
    d2 = upload(client, sample_pdfs["v1.1"], pid, "1.1")
    return pid, d1, d2


def test_full_flow(client, loaded):
    pid, d1, d2 = loaded
    st = client.get(f"/projects/{pid}/documents/{d1}/status").json()  # background task already ran in TestClient
    assert st["status"] == "ready" and st["n_pages"] == 7 and st["n_entities"] == 34, st

    r = client.post(f"/projects/{pid}/ask", headers=VIEWER, json={"question": "Which ports does the PowerManager provide?"})
    assert r.status_code == 200 and r.json()["sources"] and not r.json()["not_found"]

    ents = client.get(f"/projects/{pid}/entities", params={"type": "component", "doc": d1}).json()
    assert len(ents) == 6 and all(e["review_status"] == "pending" for e in ents)
    assert len(client.get(f"/projects/{pid}/entities", params={"status": "approved"}).json()) == 0

    f = client.get(f"/projects/{pid}/checks", params={"doc": d1}).json()
    assert {x["rule_id"] for x in f} == {"R001", "R002", "R003"}

    cmp_ = client.post(f"/projects/{pid}/compare", json={"doc_a": d1, "doc_b": d2}).json()
    assert cmp_["summary"] == {"added": 5, "removed": 1, "changed": 0, "renamed": 1}

    g = client.get(f"/projects/{pid}/graph", params={"doc": d1}).json()
    assert g["nodes"] and g["edges"] and "<html" in g["html"].lower()

    csv_ = client.get(f"/projects/{pid}/export", params={"format": "csv", "doc": d1})
    assert csv_.status_code == 200 and csv_.text.splitlines()[0].startswith("type,name") and "DoorLockManager" in csv_.text
    js = client.get(f"/projects/{pid}/export", params={"format": "json"}).json()
    assert set(js["entities"]) == {"component", "interface", "port", "signal", "dependency"} and js["findings"]

    audit = client.get(f"/projects/{pid}/audit", headers=ADMIN).json()
    actions = {a["action"] for a in audit}
    assert {"create_project", "upload", "ask", "compare", "export"} <= actions
    ask_row = next(a for a in audit if a["action"] == "ask")
    assert ask_row["user"] == "guest" and ask_row["query"] and ask_row["sources"]


def test_viewer_cannot_mutate(client, loaded, sample_pdfs):
    pid, d1, _ = loaded
    assert client.post("/projects", headers=VIEWER, json={"name": "x"}).status_code == 403
    with open(sample_pdfs["v1.0"], "rb") as f:
        assert client.post(f"/projects/{pid}/upload", headers=VIEWER, files={"file": ("a.pdf", f, "application/pdf")}).status_code == 403
    eid = client.get(f"/projects/{pid}/entities").json()[0]["id"]
    assert client.patch(f"/entities/{eid}", headers=VIEWER, json={"review_status": "approved"}).status_code == 403
    assert client.get(f"/projects/{pid}/audit", headers=VIEWER).status_code == 403
    assert client.get("/projects", headers={"X-Role": "root"}).status_code == 200  # role only validated where used
    assert client.post(f"/projects/{pid}/ask", headers={"X-Role": "root"}, json={"question": "hello there"}).status_code == 400


def test_edit_approve_and_effects(client, loaded):
    pid, d1, _ = loaded
    sig = next(e for e in client.get(f"/projects/{pid}/entities", params={"type": "signal", "doc": d1}).json()
               if e["name"] == "DiagSessionState")
    r = client.patch(f"/entities/{sig['id']}", headers=ADMIN, json={"attrs": {"receiver": "ComSignalRouter"},
                                                                   "review_status": "approved"})
    assert r.status_code == 200 and r.json()["edited"] and r.json()["review_status"] == "approved"
    f = client.get(f"/projects/{pid}/checks", params={"doc": d1}).json()
    assert "R002" not in {x["rule_id"] for x in f}  # human fix removes the finding

    port = next(e for e in client.get(f"/projects/{pid}/entities", params={"type": "port", "doc": d1}).json()
                if e["name"] == "DiagnosticHandler.DoorLockDiag")
    assert client.patch(f"/entities/{port['id']}", headers=ADMIN, json={"attrs": {"direction": "sideways"}}).status_code == 422
    assert client.patch(f"/entities/{port['id']}", headers=ADMIN, json={"name": "x"}).status_code == 422
    comp = client.get(f"/projects/{pid}/entities", params={"type": "component", "doc": d1}).json()
    clash = client.patch(f"/entities/{comp[0]['id']}", headers=ADMIN, json={"name": comp[1]["name"].lower()})
    assert clash.status_code == 409
    assert client.patch("/entities/99999", headers=ADMIN, json={"review_status": "approved"}).status_code == 404


def test_upload_validation(client):
    pid = client.post("/projects", headers=ADMIN, json={"name": "P"}).json()["id"]
    up = lambda name, body: client.post(f"/projects/{pid}/upload", headers=ADMIN, files={"file": (name, body)})  # noqa: E731
    assert up("notes.txt", b"hi").status_code == 415
    assert up("fake.pdf", b"not really").status_code == 415
    assert client.post("/projects/999/upload", headers=ADMIN, files={"file": ("a.pdf", b"%PDF-")}).status_code == 404


def test_corrupt_pdf_reports_failed_status(client):
    pid = client.post("/projects", headers=ADMIN, json={"name": "P"}).json()["id"]
    r = client.post(f"/projects/{pid}/upload", headers=ADMIN, files={"file": ("bad.pdf", b"%PDF-1.4 junk")})
    st = client.get(f"/projects/{pid}/documents/{r.json()['id']}/status").json()
    assert st["status"] == "failed" and st["message"]


def test_empty_project_errors_are_clear(client):
    pid = client.post("/projects", headers=ADMIN, json={"name": "Empty"}).json()["id"]
    assert client.post(f"/projects/{pid}/ask", json={"question": "anything?"}).status_code == 409
    assert client.get(f"/projects/{pid}/graph").status_code == 409
    assert client.get(f"/projects/{pid}/checks").json() == []
    assert client.get("/projects/404/checks").status_code == 404
    assert client.post("/projects", headers=ADMIN, json={"name": "Empty"}).status_code == 409
    assert client.post(f"/projects/{pid}/compare", json={"doc_a": 1, "doc_b": 2}).status_code == 404
