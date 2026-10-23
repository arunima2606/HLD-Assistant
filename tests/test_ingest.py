from app.ingest import build_chunks, ingest_pdf, parse_pdf


def test_sections_and_pages(sample_pdfs):
    parsed = parse_pdf(sample_pdfs["v1.0"])
    assert parsed.n_pages == 7
    pages = {s.label: s.page_start for s in parsed.sections}
    # top-level headings without body text of their own (1, 2, 3, 4) are not content sections
    expected = {"1.1 Purpose": 1, "2.1 Architecture Overview": 2, "2.2 Component Catalogue": 2,
                "3.1 Interfaces": 3, "3.2 Ports": 4, "3.2.1 Power State Handling": 4,
                "4.1 Signal List": 5, "4.2 Component Dependencies": 5, "5 Diagnostics": 6, "6 Revision History": 7}
    for label, page in expected.items():
        assert pages.get(label) == page, (label, pages)


def test_tables_become_markdown_chunks(sample_pdfs):
    _, chunks = ingest_pdf(sample_pdfs["v1.0"], "1.0")
    tables = [c for c in chunks if c.type == "table"]
    assert len(tables) == 6  # catalogue, interfaces, ports, signals, deps, revisions
    ports = next(c for c in tables if c.section.startswith("3.2 Ports"))
    assert ports.page_start == 4
    assert "| DoorLockManager | DoorLockCmd | provided | IDoorLock |" in ports.text
    assert ports.text.splitlines()[0].startswith("| Component | Port | Direction | Interface |")


def test_text_not_duplicated_from_tables(sample_pdfs):
    _, chunks = ingest_pdf(sample_pdfs["v1.0"], "1.0")
    text_chunks = [c for c in chunks if c.type == "text"]
    assert not any("DoorLockCmd" in c.text for c in text_chunks)
    assert all(c.doc == "sample_hld_v1.0.pdf" and c.version == "1.0" for c in chunks)
    assert chunks[0].embed_text.startswith(chunks[0].section)


def test_long_sections_are_split_with_overlap(sample_pdfs):
    parsed = parse_pdf(sample_pdfs["v1.0"])
    small = build_chunks(parsed, "x.pdf", "1", max_words=15, overlap=3)
    big = build_chunks(parsed, "x.pdf", "1", max_words=350, overlap=40)
    assert len(small) > len(big)
    assert all(len(c.text.split()) <= 15 + 3 + 2 for c in small if c.type == "text")


def test_bad_pdf_raises_clear_error(tmp_path):
    import pytest
    from app.ingest import IngestError
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(IngestError):
        parse_pdf(bad)
