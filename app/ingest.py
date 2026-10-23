"""PDF -> pages -> sections -> chunks (text + tables), keeping page numbers for citations."""
from __future__ import annotations

import io
import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
import pymupdf

from app.config import get_settings
from app.textutil import clean, word_count

log = logging.getLogger(__name__)

HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,5})\.?\s+([A-Za-z].{0,118})$")
MIN_PAGE_CHARS = 30


class IngestError(Exception):
    """Raised when a PDF cannot be parsed."""


@dataclass
class Block:
    kind: str  # "text" | "table"
    text: str
    page: int
    y: float = 0.0


@dataclass
class Section:
    number: str
    title: str
    page_start: int
    blocks: list[Block] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.number} {self.title}".strip()


@dataclass
class Chunk:
    text: str
    doc: str
    version: str
    section: str
    page_start: int
    page_end: int
    type: str  # "text" | "table"
    index: int = 0

    @property
    def embed_text(self) -> str:
        """Section title is prepended so retrieval sees the topic context."""
        prefix = f"{self.section}\n" if self.section else ""
        return f"{prefix}{'Table: ' if self.type == 'table' else ''}{self.text}"


@dataclass
class ParsedDocument:
    filename: str
    n_pages: int
    sections: list[Section]
    ocr_pages: list[int] = field(default_factory=list)


def rows_to_markdown(rows: list[list[str]]) -> str:
    """Render table rows (first row = header) as a markdown table."""
    width = max(len(r) for r in rows)
    norm = [[c.replace("|", "/") for c in (r + [""] * (width - len(r)))] for r in rows]
    lines = ["| " + " | ".join(norm[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in norm[1:]]
    return "\n".join(lines)


def _ocr_page(page: pymupdf.Page) -> str:
    """Optional OCR; returns '' if pytesseract / Tesseract is unavailable."""
    try:
        import pytesseract
        from PIL import Image

        pix = page.get_pixmap(dpi=200)
        return pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png"))))
    except Exception as exc:  # noqa: BLE001 - OCR is best effort
        log.warning("OCR unavailable or failed on page %s: %s", page.number + 1, exc)
        return ""


def _body_font_size(page_dicts: list[dict]) -> float:
    sizes: Counter[float] = Counter()
    for d in page_dicts:
        for b in d["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    sizes[round(s["size"], 1)] += len(s["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _in_boxes(bbox: tuple, boxes: list[tuple]) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(b[0] - 1 <= cx <= b[2] + 1 and b[1] - 1 <= cy <= b[3] + 1 for b in boxes)


def _heading(text: str, size: float, bold: bool, body: float) -> tuple[str, str] | None:
    """Return (number, title) if the line looks like a heading."""
    emphasised = bold or size >= body + 0.8
    if not emphasised or len(text) > 120:
        return None
    m = HEADING_RE.match(text)
    if m:
        return m.group(1), m.group(2).strip()
    if size >= body + 4 and not text.endswith((".", ":", ",")):  # unnumbered big title
        return "", text
    return None


def parse_pdf(path: str | Path, ocr: bool | None = None) -> ParsedDocument:
    """Parse a PDF into sections made of text and table blocks (all with page numbers)."""
    path = Path(path)
    ocr = get_settings().ocr_enabled if ocr is None else ocr
    try:
        doc = pymupdf.open(path)
        plumber = pdfplumber.open(path)
    except Exception as exc:  # noqa: BLE001
        raise IngestError(f"Cannot open PDF '{path.name}': {exc}") from exc
    if len(doc) == 0:
        raise IngestError(f"PDF '{path.name}' has no pages")

    sections: list[Section] = []
    ocr_pages: list[int] = []
    current = Section("", "", 1)
    sections.append(current)

    try:
        page_dicts = [p.get_text("dict") for p in doc]
        body = _body_font_size(page_dicts)

        for pno, (page, pdict) in enumerate(zip(doc, page_dicts), start=1):
            items: list[tuple[float, str, object]] = []  # (y, kind, payload)

            # --- tables (pdfplumber) ---
            boxes: list[tuple] = []
            try:
                for t in plumber.pages[pno - 1].find_tables():
                    rows = [[clean(c) if c is not None else "" for c in r] for r in t.extract()]
                    rows = [r for r in rows if any(r)]
                    if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                        boxes.append(t.bbox)
                        items.append((t.bbox[1], "table", rows))
            except Exception as exc:  # noqa: BLE001
                log.warning("Table extraction failed on page %d: %s", pno, exc)

            # --- text lines outside tables; paragraphs follow PyMuPDF blocks ---
            for b in pdict["blocks"]:
                if b.get("type") != 0:
                    continue
                par: list[str] = []
                par_y = b["bbox"][1]
                for line in b["lines"]:
                    text = "".join(s["text"] for s in line["spans"]).strip()
                    if not text or _in_boxes(line["bbox"], boxes):
                        continue
                    size = max(s["size"] for s in line["spans"])
                    bold = any((s["flags"] & 16) or "bold" in s["font"].lower() for s in line["spans"])
                    h = _heading(text, size, bold, body)
                    if h:
                        if par:
                            items.append((par_y, "text", " ".join(par)))
                            par = []
                        items.append((line["bbox"][1], "heading", h))
                        par_y = line["bbox"][3]
                    else:
                        if not par:
                            par_y = line["bbox"][1]
                        par.append(text)
                if par:
                    items.append((par_y, "text", " ".join(par)))

            # --- OCR fallback for (near-)empty pages ---
            if not items and len(page.get_text().strip()) < MIN_PAGE_CHARS and ocr:
                txt = _ocr_page(page)
                if txt.strip():
                    ocr_pages.append(pno)
                    for i, ln in enumerate(l.strip() for l in txt.splitlines() if l.strip()):
                        m = HEADING_RE.match(ln)
                        items.append((float(i), "heading" if m and len(ln) < 80 else "text",
                                      (m.group(1), m.group(2)) if m and len(ln) < 80 else ln))

            for _, kind, payload in sorted(items, key=lambda x: x[0]):
                if kind == "heading":
                    number, title = payload  # type: ignore[misc]
                    current = Section(number, title, pno)
                    sections.append(current)
                elif kind == "table":
                    current.blocks.append(Block("table", rows_to_markdown(payload), pno))  # type: ignore[arg-type]
                else:
                    current.blocks.append(Block("text", str(payload), pno))
        n_pages = len(doc)
    finally:
        plumber.close()
        doc.close()

    sections = [s for s in sections if s.blocks]
    return ParsedDocument(path.name, n_pages, sections, ocr_pages)


def _split_words(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    out, start = [], 0
    while start < len(words):
        out.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break
        start += max_words - overlap
    return out


def _split_table(md: str, max_words: int) -> list[str]:
    lines = md.splitlines()
    head, rows = lines[:2], lines[2:]
    out, cur = [], []
    for r in rows:
        if cur and word_count("\n".join(head + cur + [r])) > max_words * 1.5:
            out.append("\n".join(head + cur))
            cur = []
        cur.append(r)
    out.append("\n".join(head + cur))
    return out


def build_chunks(parsed: ParsedDocument, doc_name: str, version: str,
                 max_words: int | None = None, overlap: int | None = None) -> list[Chunk]:
    """Chunk by section (~max_words each); every table becomes its own chunk(s)."""
    s = get_settings()
    max_words = max_words or s.chunk_max_words
    overlap = s.chunk_overlap_words if overlap is None else overlap
    chunks: list[Chunk] = []

    def emit(text: str, sec: Section, p0: int, p1: int, typ: str) -> None:
        chunks.append(Chunk(text, doc_name, version, sec.label, p0, p1, typ, index=len(chunks)))

    for sec in parsed.sections:
        buf: list[str] = []
        pages: list[int] = []

        def flush(carry: bool = False, sec: Section = sec) -> None:
            nonlocal buf, pages
            if not buf:
                return
            text = "\n\n".join(buf)
            emit(text, sec, min(pages), max(pages), "text")
            if carry and overlap:
                tail = " ".join(text.split()[-overlap:])
                buf, pages = [tail], [pages[-1]]
            else:
                buf, pages = [], []

        for blk in sec.blocks:
            if blk.kind == "table":
                flush()
                for part in _split_table(blk.text, max_words):
                    emit(part, sec, blk.page, blk.page, "table")
                continue
            for piece in _split_words(blk.text, max_words, overlap) if word_count(blk.text) > max_words else [blk.text]:
                if buf and sum(word_count(b) for b in buf) + word_count(piece) > max_words:
                    flush(carry=True)
                buf.append(piece)
                pages.append(blk.page)
        # drop a trailing chunk that is only overlap carry-over
        if buf and not (len(buf) == 1 and chunks and chunks[-1].text.endswith(buf[0]) and chunks[-1].section == sec.label):
            flush()
    return chunks


def ingest_pdf(path: str | Path, version: str = "1.0") -> tuple[ParsedDocument, list[Chunk]]:
    parsed = parse_pdf(path)
    chunks = build_chunks(parsed, Path(path).name, version)
    if not chunks:
        raise IngestError(f"No extractable text found in '{Path(path).name}' (scanned? install Tesseract for OCR)")
    return parsed, chunks
