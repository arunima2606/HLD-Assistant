"""Embeddings + persistent ChromaDB (one collection per project = project isolation)."""
from __future__ import annotations

import logging
import threading
import zlib
from typing import Any, Protocol

import chromadb
import numpy as np

from app.config import get_settings
from app.ingest import Chunk
from app.textutil import split_identifier

log = logging.getLogger(__name__)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class IndexError_(Exception):
    """Raised for indexing/retrieval configuration problems."""


class Embedder(Protocol):
    name: str

    def encode_documents(self, texts: list[str]) -> list[list[float]]: ...
    def encode_query(self, text: str) -> list[float]: ...


_STOP = set("a an the of to in on at for and or is are was were be been what which who whom whose when where why how "
            "does do did this that these those it its with by from as into than then there their can could should "
            "would will shall may might about over under per via each all any".split())


class HashEmbedder:
    """Dependency-free lexical embedder (signed feature hashing of word uni/bi-grams).

    Offline fallback for tests / tiny machines. Quality is far below BGE/E5 -
    use EMBEDDING_BACKEND=st for real work.
    """

    name = "hash-1024"
    dim = 1024

    @staticmethod
    def _stem(word: str) -> str:
        """Crude stemmer (strip plural 's', truncate to 6 chars): depend/depends/dependencies -> 'depend'."""
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        return word[:6]

    def _tokens(self, text: str) -> list[str]:
        toks: list[str] = []
        for raw in text.split():
            words = [self._stem(w) for w in split_identifier(raw) if w not in _STOP]
            toks += words
            if len(words) > 1:
                toks.append("".join(words))  # keep CamelCase identifier as one feature as well
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for t in self._tokens(text):
            h = zlib.crc32(t.encode())
            v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        n = np.linalg.norm(v)
        return (v / n).tolist() if n else v.tolist()

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._vec(text)


class STEmbedder:
    """sentence-transformers embedder (BGE / E5 / any ST model)."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise IndexError_("sentence-transformers is not installed. `pip install sentence-transformers` "
                              "or set EMBEDDING_BACKEND=hash for the offline fallback.") from exc
        self.name = model_name
        self.model = SentenceTransformer(model_name)
        low = model_name.lower()
        self.q_prefix = BGE_QUERY_PREFIX if "bge" in low and "en" in low else ("query: " if "e5" in low else "")
        self.d_prefix = "passage: " if "e5" in low else ""

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode([self.d_prefix + t for t in texts], normalize_embeddings=True,
                                 batch_size=32).tolist()

    def encode_query(self, text: str) -> list[float]:
        return self.model.encode([self.q_prefix + text], normalize_embeddings=True)[0].tolist()


# Creation of embedders / Chroma clients must be thread-safe: FastAPI runs background tasks and requests in
# worker threads, and chromadb's shared-system init races when two threads create a client at once.
_lock = threading.RLock()
_embedders: dict[tuple[str, str], Embedder] = {}
_clients: dict[str, chromadb.ClientAPI] = {}


def get_embedder() -> Embedder:
    s = get_settings()
    key = (s.embedding_backend, s.embedding_model)
    with _lock:
        if key not in _embedders:
            _embedders[key] = HashEmbedder() if s.embedding_backend == "hash" else STEmbedder(s.embedding_model)
        return _embedders[key]


def _client(path: str) -> chromadb.ClientAPI:
    with _lock:
        if path not in _clients:
            _clients[path] = chromadb.PersistentClient(path=path)
        return _clients[path]


def collection_name(project_id: int) -> str:
    return f"proj_{project_id}"


def get_collection(project_id: int, create: bool = True):
    """Get the project's collection; guards against mixing embedding models."""
    client = _client(str(get_settings().chroma_dir))
    emb = get_embedder()
    name = collection_name(project_id)
    if create:
        col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine", "embedder": emb.name})
    else:
        try:
            col = client.get_collection(name)
        except Exception:  # noqa: BLE001 - chroma raises different errors by version
            return None
    stored = (col.metadata or {}).get("embedder")
    if stored and stored != emb.name:
        raise IndexError_(f"Project {project_id} was indexed with embedder '{stored}' but the current one is "
                          f"'{emb.name}'. Re-upload documents or restore the original EMBEDDING_* settings.")
    return col


def index_chunks(project_id: int, doc_id: int, chunks: list[Chunk]) -> list[str]:
    """Embed and store chunks; returns their Chroma ids."""
    col = get_collection(project_id)
    emb = get_embedder()
    ids = [f"d{doc_id}_c{c.index}" for c in chunks]
    for i in range(0, len(chunks), 64):
        batch = chunks[i:i + 64]
        col.upsert(
            ids=ids[i:i + 64],
            documents=[c.text for c in batch],
            embeddings=emb.encode_documents([c.embed_text for c in batch]),
            metadatas=[{"doc_id": doc_id, "doc": c.doc, "version": c.version, "section": c.section,
                        "page_start": c.page_start, "page_end": c.page_end, "type": c.type,
                        "chunk_index": c.index} for c in batch],
        )
    return ids


def delete_document(project_id: int, doc_id: int) -> None:
    col = get_collection(project_id, create=False)
    if col is not None:
        col.delete(where={"doc_id": doc_id})


def query(project_id: int, text: str, k: int = 6, doc_id: int | None = None) -> list[dict[str, Any]]:
    """Top-k semantic hits: [{text, meta, similarity}] (cosine similarity, higher = better)."""
    col = get_collection(project_id, create=False)
    if col is None or col.count() == 0:
        return []
    res = col.query(query_embeddings=[get_embedder().encode_query(text)], n_results=min(k, col.count()),
                    where={"doc_id": doc_id} if doc_id is not None else None)
    return [{"text": d, "meta": m, "similarity": round(1.0 - float(dist), 4)}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
