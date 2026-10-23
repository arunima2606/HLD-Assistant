"""End-to-end UI smoke test: real uvicorn server + headless Streamlit AppTest (MOCK_LLM, hash embeddings).

Usage: python -m scripts.ui_smoke
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    from scripts.make_sample_hld import generate

    tmp = Path(tempfile.mkdtemp(prefix="hld_ui_"))
    samples = generate(tmp / "samples")
    port = free_port()
    env = {**os.environ, "MOCK_LLM": "1", "EMBEDDING_BACKEND": "hash", "DATA_DIR": str(tmp / "data"), "PYTHONPATH": str(ROOT)}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.api:app", "--port", str(port), "--log-level", "warning"],
                              cwd=ROOT, env=env)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            try:
                requests.get(f"{base}/health", timeout=1)
                break
            except requests.RequestException:
                time.sleep(0.5)
        else:
            print("server did not start")
            return 1

        admin = {"X-Role": "admin", "X-User": "smoke"}
        pid = requests.post(f"{base}/projects", headers=admin, json={"name": "BCM"}).json()["id"]
        for key, ver in (("v1.0", "1.0"), ("v1.1", "1.1")):
            with open(samples[key], "rb") as f:
                r = requests.post(f"{base}/projects/{pid}/upload", headers=admin, data={"version": ver},
                                  files={"file": (samples[key].name, f, "application/pdf")})
            assert r.status_code == 202, r.text
        for _ in range(60):  # real background tasks: poll until both are ready
            st = [d["status"] for d in requests.get(f"{base}/projects/{pid}/documents").json()]
            if all(s in ("ready", "failed") for s in st):
                break
            time.sleep(0.5)
        assert st == ["ready", "ready"], st
        print("API seeded: 2 documents ready")

        os.environ["API_BASE_URL"] = base
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "ui" / "streamlit_app.py"), default_timeout=90).run()
        assert not at.exception, at.exception
        assert len(at.tabs) >= 8, len(at.tabs)  # 8 main tabs + nested inventory sub-tabs
        print(f"viewer render OK: {len(at.tabs)} tabs (incl. nested), no exceptions")

        at.sidebar.selectbox[0].set_value("admin").run()  # role selectbox
        assert not at.exception, at.exception
        at.chat_input[0].set_value("Which ports does the PowerManager provide?").run()
        assert not at.exception, at.exception
        chat = [m.markdown[0].value for m in at.chat_message if m.markdown]
        assert len(chat) == 2 and "PowerManager" in chat[1], chat
        print("chat OK:", chat[1][:90].replace("\n", " "))

        assert at.dataframe, "expected dataframes (checks/compare/documents)"
        editors = [e for e in at.get("arrow_data_frame")]
        print(f"rendered {len(at.dataframe)} dataframes, {len(at.get('arrow_data_frame'))} arrow elements")
        assert not at.error, [e.value for e in at.error]

        # click Compare, then check results rendered
        cmp_btn = next(b for b in at.button if b.label == "Compare")
        cmp_btn.click().run()
        assert not at.exception, at.exception
        metrics = {m.label: m.value for m in at.metric}
        assert metrics.get("Added") == "5" and metrics.get("Renamed") == "1", metrics
        print("compare OK:", {k: metrics[k] for k in ("Added", "Removed", "Changed", "Renamed")})
        print("high/medium/low findings:", metrics.get("High"), metrics.get("Medium"), metrics.get("Low"))

        prep = next(b for b in at.button if b.label == "Prepare export")
        prep.click().run()
        assert not at.exception and not at.error, at.exception
        print("export prepared OK")
        print("UI SMOKE TEST PASSED")
        return 0
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
