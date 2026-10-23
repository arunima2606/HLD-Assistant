"""Streamlit engineering UI. Talks to the FastAPI backend over HTTP (API_BASE_URL)."""
from __future__ import annotations

import base64
import os
import time

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
TYPES = ["component", "interface", "port", "signal", "dependency"]
ATTR_COLS = {"component": ["description"], "interface": ["description"],
             "port": ["component", "port", "direction", "interface"], "signal": ["sender", "receiver"],
             "dependency": ["from", "to", "dep_type"]}
SEV_COLOR = {"high": "#f8d7da", "medium": "#fff3cd", "low": "#e2e3e5"}
CONF_COLOR = {"high": "green", "medium": "orange", "low": "red", "none": "gray"}

st.set_page_config(page_title="AUTOSAR HLD Assistant", page_icon="🚗", layout="wide")


# ---------------- backend helper ----------------
def call(method: str, path: str, *, raw: bool = False, quiet: bool = False, **kw):
    """HTTP call with role headers. Shows a readable error and returns None on failure."""
    headers = {"X-Role": st.session_state.get("role", "viewer"), "X-User": st.session_state.get("user", "anonymous")}
    try:
        r = requests.request(method, f"{API}{path}", headers=headers, timeout=kw.pop("timeout", 120), **kw)
    except requests.RequestException:
        if not quiet:
            st.error(f"Cannot reach the API at {API}. Start it with `make api`.")
        return None
    if r.status_code >= 400:
        if not quiet:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            st.error(f"{r.status_code}: {detail}")
        return None
    return r if raw else r.json()


def is_admin() -> bool:
    return st.session_state.get("role") == "admin"


def doc_label(d: dict) -> str:
    return f"{d['filename']}  v{d['version']}  (#{d['id']}, {d['status']})"


def pick_doc(docs: list[dict], key: str, label: str = "Document", ready_only: bool = True):
    pool = [d for d in docs if d["status"] == "ready"] if ready_only else docs
    if not pool:
        st.info("No processed documents yet - upload one in the Upload tab.")
        return None
    return st.selectbox(label, pool, index=len(pool) - 1, format_func=doc_label, key=key)


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("🚗 HLD Assistant")
    health = call("GET", "/health", quiet=True)
    if health:
        st.caption(f"API ✅ · LLM: {'MOCK' if health['mock_llm'] else health['llm_model']} · embeddings: {health['embedding_backend']}")
    else:
        st.caption(f"API ❌ not reachable at {API}")
    st.text_input("Your name", key="user", value=st.session_state.get("user", "engineer"))
    st.selectbox("Role", ["viewer", "admin"], key="role",
                 help="Pilot-level stand-in for real RBAC. Viewers are read-only.")
    projects = call("GET", "/projects") or []
    project = st.selectbox("Project", projects, format_func=lambda p: p["name"]) if projects else None
    with st.expander("➕ New project", expanded=not projects):
        new_name = st.text_input("Project name", key="new_project")
        if st.button("Create project", disabled=not is_admin() or not new_name.strip()):
            if call("POST", "/projects", json={"name": new_name.strip()}):
                st.rerun()
        if not is_admin():
            st.caption("Switch role to **admin** to create projects or upload.")
    st.divider()
    st.caption("⚠️ AI output is advisory. Review and approve entities before using them for design decisions.")

if not project:
    st.info("Create a project in the sidebar to begin.")
    st.stop()

pid = project["id"]
docs = call("GET", f"/projects/{pid}/documents") or []
tabs = st.tabs(["📤 Upload", "💬 Ask", "📋 Inventory", "🕸️ Dependency Map", "✅ Checks", "🔀 Compare", "⬇️ Export", "🧾 Audit"])

# ---------------- Upload ----------------
with tabs[0]:
    st.subheader("Upload an HLD document")
    up = st.file_uploader("PDF", type=["pdf"], disabled=not is_admin())
    version = st.text_input("Version label", "1.0")
    if st.button("Process document", type="primary", disabled=not (is_admin() and up)):
        doc = call("POST", f"/projects/{pid}/upload", files={"file": (up.name, up.getvalue(), "application/pdf")},
                   data={"version": version})
        if doc:
            box = st.empty()
            for _ in range(600):
                s = call("GET", f"/projects/{pid}/documents/{doc['id']}/status", quiet=True)
                if not s:
                    break
                box.info(f"⏳ {s['status']}: {s['message']}")
                if s["status"] in ("ready", "failed"):
                    (box.success if s["status"] == "ready" else box.error)(f"{s['status']}: {s['message']}")
                    break
                time.sleep(1.5)
    if docs:
        st.dataframe(pd.DataFrame(docs)[["id", "filename", "version", "status", "n_pages", "n_chunks", "n_entities", "message"]],
                     hide_index=True, width="stretch")
        if st.button("🔄 Refresh status"):
            st.rerun()

# ---------------- Ask ----------------
with tabs[1]:
    st.subheader("Ask the documents")
    ready = [d for d in docs if d["status"] == "ready"]
    scope = st.selectbox("Search in", [None] + ready, format_func=lambda d: "All documents" if d is None else doc_label(d))
    hist_key = f"chat_{pid}"
    history = st.session_state.setdefault(hist_key, [])
    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("resp"):
                r = m["resp"]
                if r["not_found"]:
                    st.warning("Nothing relevant was found in the indexed documents.")
                else:
                    st.markdown(f"Confidence: :{CONF_COLOR[r['confidence_label']]}[**{r['confidence_label']}** ({r['confidence']})]"
                                " - retrieval match, not a guarantee of correctness")
                    with st.expander(f"Sources ({len(r['sources'])})"):
                        for s in r["sources"]:
                            pages = s["page_start"] if s["page_start"] == s["page_end"] else f"{s['page_start']}-{s['page_end']}"
                            st.markdown(f"**{s['doc']}** v{s['version']} · {s['section']} · p.{pages} · {s['type']} · sim {s['similarity']}")
                            st.caption(s["snippet"])
    if q := st.chat_input("e.g. Which components require the IPowerState interface?", disabled=not ready):
        history.append({"role": "user", "content": q})
        with st.spinner("Searching and generating…"):
            resp = call("POST", f"/projects/{pid}/ask", json={"question": q, "doc_id": scope["id"] if scope else None}, timeout=300)
        history.append({"role": "assistant", "content": resp["answer"] if resp else "_Request failed - see error above._", "resp": resp})
        st.rerun()
    if not ready:
        st.info("Upload and process a document first.")

# ---------------- Inventory ----------------
with tabs[2]:
    st.subheader("Architecture inventory (human review)")
    doc = pick_doc(docs, "inv_doc")
    if doc:
        flt = st.radio("Show", ["all", "pending", "approved", "rejected"], horizontal=True)
        ents = call("GET", f"/projects/{pid}/entities", params={"doc": doc["id"]}) or []
        counts = pd.Series([e["review_status"] for e in ents]).value_counts()
        st.caption("  ·  ".join(f"{k}: {counts.get(k, 0)}" for k in ["pending", "approved", "rejected"]))
        sub = st.tabs([f"{t}s ({sum(e['type'] == t for e in ents)})" for t in TYPES])
        for t, tab in zip(TYPES, sub):
            with tab:
                rows = [e for e in ents if e["type"] == t and (flt == "all" or e["review_status"] == flt)]
                if not rows:
                    st.caption("Nothing to show.")
                    continue
                cols = ATTR_COLS[t]
                df = pd.DataFrame([{"id": e["id"], "name": e["name"], **{c: e["attrs"].get(c, "") for c in cols},
                                    "pages": ",".join(str(p) for p in sorted({s["page"] for s in e["sources"]})),
                                    "status": e["review_status"]} for e in rows])
                locked = ["id", "pages"] + (["name"] if t in ("port", "dependency") else [])
                edited = st.data_editor(df, key=f"ed_{t}_{doc['id']}_{flt}", hide_index=True, width="stretch",
                                        disabled=True if not is_admin() else locked,
                                        column_config={"status": st.column_config.SelectboxColumn(options=["pending", "approved", "rejected"], required=True)})
                c1, c2 = st.columns(2)
                if c1.button("💾 Save changes", key=f"save_{t}", disabled=not is_admin()):
                    n = 0
                    for old, new in zip(df.to_dict("records"), edited.to_dict("records")):
                        patch: dict = {}
                        if new["status"] != old["status"]:
                            patch["review_status"] = new["status"]
                        attrs = {c: new[c] for c in cols if new[c] != old[c]}
                        if attrs:
                            patch["attrs"] = attrs
                        if t not in ("port", "dependency") and new["name"] != old["name"]:
                            patch["name"] = new["name"]
                        if patch and call("PATCH", f"/entities/{old['id']}", json=patch):
                            n += 1
                    st.success(f"Saved {n} change(s).")
                    time.sleep(0.6)
                    st.rerun()
                if c2.button(f"✅ Approve all pending {t}s", key=f"appr_{t}", disabled=not is_admin()):
                    for e in rows:
                        if e["review_status"] == "pending":
                            call("PATCH", f"/entities/{e['id']}", json={"review_status": "approved"})
                    st.rerun()

# ---------------- Dependency map ----------------
with tabs[3]:
    st.subheader("Dependency map")
    doc = pick_doc(docs, "graph_doc")
    if doc:
        g = call("GET", f"/projects/{pid}/graph", params={"doc": doc["id"]})
        if g:
            st.caption("🟦 component · 🟧 interface · 🟥 referenced but undefined · green edge = signal · dashed = required port")
            # data: URL => opaque origin: the graph page cannot touch the Streamlit app, whatever the labels contain.
            st.iframe("data:text/html;base64," + base64.b64encode(g["html"].encode()).decode(), height=640)
            st.caption(f"{len(g['nodes'])} nodes · {len(g['edges'])} edges (rejected entities are excluded)")

# ---------------- Checks ----------------
with tabs[4]:
    st.subheader("Consistency checks (deterministic rules)")
    scope = st.selectbox("Document", [None] + [d for d in docs if d["status"] == "ready"],
                         format_func=lambda d: "All documents" if d is None else doc_label(d), key="chk_doc")
    found = call("GET", f"/projects/{pid}/checks", params={"doc": scope["id"]} if scope else None)
    if found is not None:
        if not found:
            st.success("No findings.")
        else:
            m = st.columns(3)
            for col, sev in zip(m, ["high", "medium", "low"]):
                col.metric(sev.capitalize(), sum(f["severity"] == sev for f in found))
            df = pd.DataFrame([{"severity": f["severity"], "rule": f["rule_id"], "version": f["version"], "message": f["message"],
                                "evidence": "; ".join(f"{c['entity']} (p.{c['page']}, {c['section']})" for c in f["citations"])}
                               for f in found])
            st.dataframe(df.style.apply(lambda r: [f"background-color: {SEV_COLOR[r['severity']]}"] * len(r), axis=1),
                         hide_index=True, width="stretch")
            st.caption("Findings are candidates for engineer review - they depend on extraction quality.")

# ---------------- Compare ----------------
with tabs[5]:
    st.subheader("Compare two versions")
    ready = [d for d in docs if d["status"] == "ready"]
    if len(ready) < 2:
        st.info("Upload and process at least two documents (e.g. v1.0 and v1.1).")
    else:
        c1, c2 = st.columns(2)
        a = c1.selectbox("Baseline (A)", ready, format_func=doc_label, key="cmp_a")
        b = c2.selectbox("Revised (B)", ready, index=len(ready) - 1, format_func=doc_label, key="cmp_b")
        if st.button("Compare", type="primary"):
            st.session_state["cmp"] = call("POST", f"/projects/{pid}/compare", json={"doc_a": a["id"], "doc_b": b["id"]})
        r = st.session_state.get("cmp")
        if r:
            sm = r["summary"]
            st.markdown(f"**v{r['doc_a']['version']} → v{r['doc_b']['version']}**")
            for col, k in zip(st.columns(4), ["added", "removed", "changed", "renamed"]):
                col.metric(k.capitalize(), sm[k])
            cite = lambda c: f"{c['doc']} v{c['version']} p.{c['page']} ({c['section']})"  # noqa: E731
            for k in ("added", "removed"):
                if r[k]:
                    with st.expander(f"{k.capitalize()} ({len(r[k])})", expanded=True):
                        st.dataframe(pd.DataFrame([{"type": i["type"], "name": i["name"], "source": cite(i["citation"])} for i in r[k]]),
                                     hide_index=True, width="stretch")
            if r["renamed"]:
                with st.expander(f"Renamed ({len(r['renamed'])})", expanded=True):
                    st.dataframe(pd.DataFrame([{"type": i["type"], "from": i["from"], "to": i["to"]} for i in r["renamed"]]),
                                 hide_index=True, width="stretch")
            if r["changed"]:
                with st.expander(f"Changed ({len(r['changed'])})", expanded=True):
                    st.dataframe(pd.DataFrame([{"type": i["type"], "name": i["name"],
                                                "changes": "; ".join(f"{f}: {v['from']!r} → {v['to']!r}" for f, v in i["changes"].items())}
                                               for i in r["changed"]]), hide_index=True, width="stretch")

# ---------------- Export ----------------
with tabs[6]:
    st.subheader("Export structured findings")
    fmt = st.radio("Format", ["csv", "json"], horizontal=True)
    scope = st.selectbox("Document", [None] + docs, format_func=lambda d: "All documents" if d is None else doc_label(d), key="exp_doc")
    if st.button("Prepare export"):
        resp = call("GET", f"/projects/{pid}/export", raw=True, params={"format": fmt, **({"doc": scope["id"]} if scope else {})})
        if resp is not None:
            st.session_state["export"] = (resp.content, fmt)
    if exp := st.session_state.get("export"):
        st.download_button(f"⬇️ Download {exp[1].upper()}", exp[0], file_name=f"{project['name']}_export.{exp[1]}",
                           mime="text/csv" if exp[1] == "csv" else "application/json")

# ---------------- Audit ----------------
with tabs[7]:
    st.subheader("Audit log")
    if not is_admin():
        st.info("Audit log is visible to admins only.")
    else:
        log = call("GET", f"/projects/{pid}/audit", params={"limit": 300}) or []
        if log:
            st.dataframe(pd.DataFrame([{"time": r["ts"], "user": r["user"], "role": r["role"], "action": r["action"],
                                        "query": r["query"], "sources": "; ".join(f"{s['doc']} p.{s['page']}" for s in r["sources"]),
                                        "details": str(r["details"])[:120]} for r in log]), hide_index=True, width="stretch")
        else:
            st.caption("No entries yet.")
