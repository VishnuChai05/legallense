import fitz  # PyMuPDF

from legal_scraper.pdf_highlighter import highlight_terms_in_pdf


def _make_pdf(tmp_path, text: str):
    path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


def _count_annots(pdf_path):
    doc = fitz.open(pdf_path)
    try:
        total = 0
        for page in doc:
            annots = list(page.annots() or [])
            total += len(annots)
        return total
    finally:
        doc.close()


def test_highlight_terms_creates_annotations(tmp_path):
    src_pdf = _make_pdf(tmp_path, "This contract has indemnity and liability clauses.")
    annotated = highlight_terms_in_pdf(src_pdf, ["indemnity", "liability"])

    assert annotated.exists()
    assert _count_annots(annotated) >= 2


def test_highlight_terms_no_crash_on_missing_term(tmp_path):
    src_pdf = _make_pdf(tmp_path, "This agreement mentions nothing else.")
    annotated = highlight_terms_in_pdf(src_pdf, ["indemnity", "liability"])

    assert annotated.exists()
    # May be zero annotations; should not raise
    assert _count_annots(annotated) >= 0