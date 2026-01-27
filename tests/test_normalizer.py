from pathlib import Path

import fitz  # PyMuPDF

from legal_scraper.gazette_model import GazetteRecord
from legal_scraper.normalizer import apply_relevance, normalize_record


def _make_pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    out = doc.tobytes()
    doc.close()
    return out


def test_normalize_record_extracts_notification_and_effective_date():
    pdf_bytes = _make_pdf_bytes("G.S.R. 123(E)\nThis shall come into force on 31-01-2024.")
    rec = normalize_record(
        source_type="Act",
        jurisdiction="India",
        state=None,
        gazette_type=None,
        title="Transfer of Property Act",
        pdf_url="http://example.com/a.pdf",
        pdf_hash_value="hash1",
        source_page_url="http://example.com",
        pdf_bytes=pdf_bytes,
    )
    assert rec.notification_number.startswith("G.S.R.")
    assert rec.date == "2024-01-31"


def test_normalize_record_falls_back_to_listing_date():
    pdf_bytes = _make_pdf_bytes("No date phrase here.")
    rec = normalize_record(
        source_type="Gazette",
        jurisdiction="State",
        state="KA",
        gazette_type="Weekly",
        title="Some Notice",
        pdf_url="http://example.com/b.pdf",
        pdf_hash_value="hash2",
        source_page_url="http://example.com",
        pdf_bytes=pdf_bytes,
        listing_date="2024-02-10",
    )
    assert rec.date == "2024-02-10"


def test_apply_relevance_act_whitelisted():
    rec = GazetteRecord(
        source_type="Act",
        jurisdiction="India",
        state=None,
        gazette_type=None,
        title="Some Act",
        notification_number="Act123",
        date="2024-01-01",
        pdf_url="http://example.com/act.pdf",
        pdf_hash="h1",
        source_page_url="http://example.com",
        contract_domain=None,
    )
    kept, filtered = apply_relevance([rec])
    assert len(kept) == 1
    assert kept[0].contract_domain == ["contract"]
    assert not filtered


def test_apply_relevance_karnataka_exclusion():
    rec = GazetteRecord(
        source_type="Gazette",
        jurisdiction="State",
        state="KA",
        gazette_type="Weekly",
        title="Seniority list of officers",
        notification_number="No.1",
        date="2024-01-01",
        pdf_url="http://example.com/ka.pdf",
        pdf_hash="h2",
        source_page_url="http://example.com",
        contract_domain=None,
    )
    kept, filtered = apply_relevance([rec])
    assert len(kept) == 0
    assert len(filtered) == 1


def test_apply_relevance_karnataka_include_property():
    rec = GazetteRecord(
        source_type="Gazette",
        jurisdiction="State",
        state="KA",
        gazette_type="Weekly",
        title="Lease agreement notification",
        notification_number="No.2",
        date="2024-01-02",
        pdf_url="http://example.com/ka2.pdf",
        pdf_hash="h3",
        source_page_url="http://example.com",
        contract_domain=None,
    )
    kept, filtered = apply_relevance([rec])
    assert len(kept) == 1
    assert "property" in kept[0].contract_domain
    assert len(filtered) == 0
