# AUTOSAR HLD Analysis Assistant

Local, RAG-based assistant that ingests AUTOSAR High-Level Design (HLD) PDFs and lets engineers:

1. **Ask** questions in natural language - answers carry `[doc | section | page]` citations and a confidence indicator
2. Get a structured **inventory** of components, interfaces, ports, signals and dependencies - each with source pages
3. See a **dependency map**
4. **Compare** two document versions at entity level (added / removed / changed / renamed)
5. Review rule-based **consistency findings**
6. **Export** CSV / JSON

> **Governance:** AI output is advisory. Every extracted entity starts as `pending`; a human must review/approve it
> before it is used for design approval, compliance, release or vehicle validation. The assistant never auto-approves
> anything and never modifies source documents. Everything runs locally - no external API calls at runtime.

## Architecture

```
 Engineer ──► Streamlit UI (ui/streamlit_app.py)
                   │  HTTP (API_BASE_URL)
                   ▼
            FastAPI  (app/api.py)   X-Role / X-User headers · audit log
        ┌──────────┼───────────────────────────────┐
        ▼          ▼                               ▼
  pipeline.py    rag.py                        checks / compare / graph
  (background)   retrieve ► prompt ► LLM       (deterministic, no LLM)
   │  │  │          │           │
   │  │  │          ▼           ▼
   │  │  │      ChromaDB     Ollama (local LLM, default qwen2.5:7b)
   │  │  │   (1 collection    MOCK_LLM=1 → deterministic canned output
   │  │  │    per project)
   │  │  └─► extract.py : table parser (deterministic) + LLM JSON (pydantic, 1 retry, grounding filter)
   │  └────► index.py   : BGE embeddings (sentence-transformers) or hash fallback ► Chroma
   └───────► ingest.py  : PyMuPDF + pdfplumber ► sections ► chunks (+ optional Tesseract OCR)
                                   │
                           SQLite (SQLAlchemy): projects, documents, chunks, entities, audit_log
```

## Quick start (local)

```bash
make setup && source .venv/bin/activate     # or: python -m venv .venv && pip install -r requirements-dev.txt
ollama pull qwen2.5:7b                      # one-off; try qwen2.5:14b / llama3.1:8b if RAM allows
make sample                                 # writes data/sample_hld_v1.0.pdf, v1.1.pdf and ground-truth JSON
make api                                    # terminal 1  → http://localhost:8000/docs
make ui                                     # terminal 2  → http://localhost:8501
```

In the UI: sidebar → role **admin** → create a project → **Upload** `sample_hld_v1.0.pdf` (version `1.0`) and
`sample_hld_v1.1.pdf` (`1.1`) → explore the tabs. The first run downloads the BGE model (~130 MB).

No Ollama yet? `MOCK_LLM=1 make api` still works end to end (answers are extractive snippets, not generated text).
No PyTorch? `EMBEDDING_BACKEND=hash` uses an offline lexical embedder (fine for demos, much weaker retrieval).

### Docker
```bash
ollama serve &                               # Ollama stays on the host
docker compose up --build                    # API :8000, UI :8501 (data in the hld_data volume)
```

## Testing & evaluation
```bash
make test     # 37 tests, MOCK_LLM + hash embeddings: no Ollama/GPU needed
make smoke    # starts a real uvicorn server and drives the Streamlit app headlessly
make eval     # precision/recall/F1 of extraction vs data/ground_truth_v1.0.json (real LLM unless MOCK_LLM=1)
python -m scripts.evaluate_extraction --pdf your.pdf --truth your_truth.json   # same format as the sample JSON
```
The sample PDFs contain **planted inconsistencies** (a required port with no provider, a signal with no receiver, a
dependency on an undefined component) and v1.1 contains a known diff (added component, renamed signal, removed port).
The tests assert all of them are found - and that nothing else is flagged.

## How the pieces work

| Concern | Approach |
|---|---|
| Parsing | PyMuPDF text with font info → numbered/bold headings → sections; pdfplumber tables → markdown chunks (text inside tables is not duplicated) |
| Chunking | By section, ~350 words max with 40-word overlap; tables are their own chunks (split with repeated header if huge); section title prepended before embedding |
| Retrieval | Cosine similarity in a per-project Chroma collection (project isolation); optional per-document filter |
| Answering | Strict prompt: answer only from context, cite labels, reply `Not found in the provided documents.` otherwise; temperature 0.1; context treated as untrusted |
| Confidence | `(top-1 similarity + mean of top-3) / 2`, labelled high/medium/low by thresholds that depend on the embedding backend. It measures **how well the question matches the retrieved text, not whether the answer is right** |
| Extraction | Recognised table headers are parsed deterministically; prose and unknown tables go to the LLM (strict JSON, pydantic validation, one retry, then the chunk is skipped and logged). LLM entities whose names don't literally occur in the source chunk are dropped |
| Merging | Names normalised (case/space/underscore/hyphen-insensitive), duplicates merged across chunks, all source pages kept |
| Checks | R001 required port without provider · R002 signal missing sender/receiver · R003 dependency on undefined component · R004 near-duplicate names · R005 isolated component · R006 port on undefined component · R007 signal endpoint undefined |
| Compare | Diff of extracted entities between two documents (not raw text); removed+added signals with identical endpoints are reported as renames |

### API (see `/docs` for schemas)
`POST /projects` · `GET /projects` · `POST /projects/{id}/upload` · `GET /projects/{id}/documents[/{doc}/status]` ·
`POST /projects/{id}/ask` · `GET /projects/{id}/entities?type=&doc=&status=` · `PATCH /entities/{id}` ·
`GET /projects/{id}/graph` · `GET /projects/{id}/checks` · `POST /projects/{id}/compare` ·
`GET /projects/{id}/export?format=csv|json` · `GET /projects/{id}/audit`

Roles: send `X-Role: admin|viewer` (default `viewer`) and `X-User: name`. Viewers can ask, browse, compare and export;
only admins can create projects, upload, edit/approve and read the audit log.

## Assumptions (made without asking)
- **Pilot RBAC:** `X-Role` is an unauthenticated header - a stand-in for real SSO/IAM. Do not expose the API to untrusted networks.
- **No `Relationship` table:** the graph, checks and compare are derived directly from `entities` (single source of truth, so
  human edits/rejections apply instantly).
- Top-level headings with no body text of their own (e.g. "4 Communication") are not stored as separate sections; their
  sub-sections carry the content.
- Documents are processed **one at a time** (a lock queues concurrent uploads) - the embedder, Chroma and SQLite are not
  built for parallel ingestion.
- Re-processing a document replaces its entities, **including review decisions**.
- The graph page is embedded via a `data:` URL iframe and all labels are HTML-escaped, because entity names come from
  uploaded documents and must be treated as untrusted.

## Known limitations
- **Diagrams/images are not parsed** (only text and ruled tables). Rely on page citations + the review queue.
- Tables without ruling lines, or tables that continue across pages, may be missed or split.
- Heading detection needs numbered headings (`3.2.1 Title`) or a clearly larger/bold unnumbered title.
- Small local models sometimes break JSON output: such chunks are skipped and counted in the document's status message.
  Try a larger model or tighten `CHUNK_MAX_WORDS`.
- The **hash embedder** is lexical only and cannot reliably reject off-topic questions; use BGE/E5 for real work.
- Confidence thresholds are untuned defaults - calibrate on your own questions.
- Extraction metrics on the synthetic sample (1.0 everywhere) show the pipeline is correct, **not** that real HLDs will
  score that well. Build ground truth for 2-3 real sections and run `make eval` with your LLM.

## Verification status
Verified in the build sandbox: all 37 tests; a real uvicorn server + headless Streamlit run (upload → background
processing → chat → inventory → checks → compare → export); parallel-upload regression; XSS-escaping regression.

**Not verified** (no GPU/Docker/browser/PyTorch available there): real Ollama answers and JSON-extraction quality,
BGE/E5 embeddings, the Docker build, Tesseract OCR, and the *visual* rendering of the graph in a browser.
Treat those as the first things to check on your machine.
