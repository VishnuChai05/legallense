import pytest

from legal_scraper.fetch import parse_html, parse_pdf, save_raw, scrape_source
from legal_scraper.sources import default_sources


def test_parse_html_simple():
    html = b"<html><body><h1>Title</h1><script>bad()</script><p>Text</p></body></html>"
    text = parse_html(html)
    assert "Title" in text and "Text" in text
    assert "bad()" not in text


def test_parse_pdf_minimal(tmp_path, monkeypatch):
    # Build a minimal PDF
    minimal_pdf = (
        b"%PDF-1.1\n"
        b"1 0 obj<<>>endobj\n"
        b"2 0 obj<< /Type /Catalog /Pages 3 0 R >>endobj\n"
        b"3 0 obj<< /Type /Pages /Kids [4 0 R] /Count 1 >>endobj\n"
        b"4 0 obj<< /Type /Page /Parent 3 0 R /MediaBox [0 0 612 792] /Contents 5 0 R >>endobj\n"
        b"5 0 obj<< /Length 12 >>stream\nBT /F1 12 Tf ET\nendstream endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000120 00000 n \n0000000200 00000 n \n0000000300 00000 n \n"
        b"trailer<< /Root 2 0 R /Size 6 >>\nstartxref\n360\n%%EOF\n"
    )
    text = parse_pdf(minimal_pdf)
    assert "BT" not in text  # ensure stream markers removed


def test_save_raw(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRAPE_DATA_DIR", str(tmp_path))
    from importlib import reload
    import legal_scraper.fetch as fetch
    reload(fetch)

    path = fetch.save_raw("Sample Doc", "http://example.com", b"content", "text/plain")
    assert path.exists()
    meta = path.with_suffix(".meta").read_text()
    assert "http://example.com" in meta


def test_default_sources_present():
    sources = default_sources()
    assert any("Contract Act" in s.name for s in sources)


def test_scrape_source_html(monkeypatch, tmp_path):
    def fake_get(url, headers=None):
        class Resp:
            status_code = 200
            headers = {"content-type": "text/html"}

            def raise_for_status(self):
                pass

            @property
            def content(self):
                return b"<html><body><p>Hello</p></body></html>"

        return Resp()

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def get(self, url):
            return fake_get(url)

    monkeypatch.setenv("SCRAPE_DATA_DIR", str(tmp_path))
    import importlib
    import legal_scraper.fetch as fetch
    importlib.reload(fetch)

    monkeypatch.setattr(fetch.httpx, "Client", DummyClient)

    text, raw_path = fetch.scrape_source("Example", "http://example.com", "html")
    assert "Hello" in text
    assert raw_path.exists()