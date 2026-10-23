import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["MOCK_LLM"] = "1"
os.environ["EMBEDDING_BACKEND"] = "hash"

from scripts.make_sample_hld import generate  # noqa: E402


@pytest.fixture(scope="session")
def sample_pdfs(tmp_path_factory):
    return generate(tmp_path_factory.mktemp("samples"))


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """Fresh DB + Chroma dir for every test."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    yield
