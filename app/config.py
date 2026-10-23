"""Environment-based settings. Read at call time so tests can override env vars."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_url: str
    chroma_dir: Path
    upload_dir: Path
    llm_model: str
    ollama_host: str
    mock_llm: bool
    llm_temperature: float
    llm_timeout: int
    embedding_backend: str  # "st" (sentence-transformers) | "hash" (offline fallback)
    embedding_model: str
    min_similarity: float
    conf_high: float
    conf_medium: float
    ocr_enabled: bool
    chunk_max_words: int
    chunk_overlap_words: int
    max_upload_mb: int
    api_base_url: str


def get_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", str(ROOT / "data"))).resolve()
    backend = os.getenv("EMBEDDING_BACKEND", "st").strip().lower()
    # Cosine-similarity scales differ per backend, so defaults differ too.
    d_min, d_high, d_med = (0.30, 0.60, 0.45) if backend == "st" else (0.12, 0.30, 0.18)
    s = Settings(
        data_dir=data_dir,
        database_url=os.getenv("DATABASE_URL", f"sqlite:///{data_dir / 'app.db'}"),
        chroma_dir=Path(os.getenv("CHROMA_DIR", str(data_dir / "chroma_db"))),
        upload_dir=Path(os.getenv("UPLOAD_DIR", str(data_dir / "uploads"))),
        llm_model=os.getenv("LLM_MODEL", "qwen2.5:7b"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        mock_llm=_flag("MOCK_LLM"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        llm_timeout=int(os.getenv("LLM_TIMEOUT", "180")),
        embedding_backend=backend,
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        min_similarity=float(os.getenv("MIN_SIMILARITY", d_min)),
        conf_high=float(os.getenv("CONF_HIGH", d_high)),
        conf_medium=float(os.getenv("CONF_MEDIUM", d_med)),
        ocr_enabled=_flag("OCR_ENABLED", "1"),
        chunk_max_words=int(os.getenv("CHUNK_MAX_WORDS", "350")),
        chunk_overlap_words=int(os.getenv("CHUNK_OVERLAP_WORDS", "40")),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "50")),
        api_base_url=os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/"),
    )
    for p in (s.data_dir, s.upload_dir, s.chroma_dir):
        p.mkdir(parents=True, exist_ok=True)
    return s


def setup_logging() -> None:
    """Configure structured-ish logging once."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
