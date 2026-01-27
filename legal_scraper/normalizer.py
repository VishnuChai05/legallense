from __future__ import annotations

import re
from datetime import datetime
from io import BytesIO
from typing import Iterable, List, Tuple

from pypdf import PdfReader

from .gazette_model import GazetteRecord

# Contract-focused keyword buckets
_KEYWORDS = {
    "contract": [
        "contract",
        "agreement",
        "vendor",
        "supplier",
        "service provider",
        "consultant",
        "franchise",
        "agency",
        "distribution",
        "outsourcing",
    ],
    "property": [
        "lease",
        "rent",
        "tenancy",
        "licensee",
        "licence",
        "lessor",
        "lessee",
        "developer",
        "allottee",
        "mortgage",
        "charge",
        "registration",
        "stamp duty",
        "rera",
        "builder",
    ],
    "employment": [
        "employment",
        "employer",
        "employee",
        "wages",
        "labour",
        "contract labour",
        "gratuity",
        "bonus",
        "consultant",
        "independent contractor",
    ],
    "regulatory": [
        "gst",
        "vat",
        "tcs",
        "tds",
        "excise",
        "customs",
        "surcharge",
        "cess",
        "property tax",
        "guidance value",
        "circle rate",
        "notification",
        "order",
        "circular",
        "scheme",
        "policy",
        "regulation",
        "rules",
        "guideline",
        "amendment",
    ],
}

STATE_EXCLUDE = {
    "KA": ["seniority", "seniority list", "promotion"],
}

STATE_INCLUDE = {
    "KA": {
        "contract": ["contract", "agreement", "vendor", "supplier", "service provider", "consultant", "tender", "bid", "ppp"],
        "property": ["lease", "rent", "tenancy", "license", "licence", "allotment", "allottee", "developer", "rera", "registration", "stamp", "guidance value", "circle rate", "land"],
        "employment": ["employment", "employer", "employee", "wages", "labour", "labor", "minimum wages", "gratuity", "bonus", "contract labour"],
    }
}

DATE_PATTERNS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%y", "%d/%m/%y"]
DATE_RE = re.compile(r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})")
NOTIF_RES = [
    re.compile(r"\bG\.S\.R\.?\s*[^\s,]+", re.IGNORECASE),
    re.compile(r"\bS\.O\.?\s*[^\s,]+", re.IGNORECASE),
    re.compile(r"\bNo\.\s*[A-Z0-9./-]+", re.IGNORECASE),
    re.compile(r"\bLAD-[A-Z]+/GN/[A-Z0-9./-]+", re.IGNORECASE),
]
EFFECTIVE_RE = [
    re.compile(r"come into force on\s+(?P<date>[\w\s.,/-]+?)\b", re.IGNORECASE),
    re.compile(r"come into effect on\s+(?P<date>[\w\s.,/-]+?)\b", re.IGNORECASE),
    re.compile(r"effective from\s+(?P<date>[\w\s.,/-]+?)\b", re.IGNORECASE),
]
DATED_RE = [
    re.compile(r"dated[:\s]+(?P<date>\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", re.IGNORECASE),
    re.compile(r"date[:\s]+(?P<date>\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", re.IGNORECASE),
]


def _normalize_date_str(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = raw.strip().replace("\u00a0", " ")
    # First try parsing the whole string with known formats
    for fmt in DATE_PATTERNS:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue

    # Extract the first date-like token if the string contains extra words
    m = DATE_RE.search(cleaned)
    candidate = m.group(1) if m else None
    if candidate:
        for fmt in DATE_PATTERNS:
            try:
                dt = datetime.strptime(candidate, fmt)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                continue
    return None


def _tag_domains(text: str) -> List[str]:
    lowered = text.lower()
    hits: List[str] = []
    for domain, keywords in _KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            hits.append(domain)
    return sorted(set(hits))


def extract_pdf_text(data: bytes, max_pages: int = 2) -> str:
    try:
        reader = PdfReader(BytesIO(data))
        texts: List[str] = []
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            texts.append(page.extract_text() or "")
        return "\n".join(texts)
    except Exception:
        return ""


def extract_notification_number(text: str, listing_hint: str | None = None) -> str | None:
    candidates: List[str] = []
    if listing_hint:
        candidates.append(listing_hint)
    for pattern in NOTIF_RES:
        m = pattern.search(text)
        if m:
            candidates.append(m.group(0))
    for cand in candidates:
        cleaned = " ".join(cand.split())
        if cleaned:
            return cleaned
    return None


def _find_first_date(text: str) -> str | None:
    """Grab the first date-like token anywhere in the text."""

    m = DATE_RE.search(text)
    if not m:
        return None
    return _normalize_date_str(m.group(1))


def extract_effective_date(text: str, listing_date: str | None = None) -> str | None:
    for pattern in EFFECTIVE_RE:
        m = pattern.search(text)
        if m:
            normalized = _normalize_date_str(m.group("date"))
            if normalized:
                return normalized
    for pattern in DATED_RE:
        m = pattern.search(text)
        if m:
            normalized = _normalize_date_str(m.group("date"))
            if normalized:
                return normalized
    # Do not default to listing_date here; upstream handles that priority.
    return _find_first_date(text)


def dedupe_records(records: Iterable[GazetteRecord]) -> List[GazetteRecord]:
    seen = set()
    deduped: List[GazetteRecord] = []
    for rec in records:
        pk = rec.primary_key() or rec.pdf_hash
        if pk in seen:
            continue
        seen.add(pk)
        deduped.append(rec)
    return deduped


def apply_relevance(records: Iterable[GazetteRecord]) -> Tuple[List[GazetteRecord], List[GazetteRecord]]:
    """Tag contract domains and filter out records with no domain match.

    Returns (kept, filtered).
    """
    kept: List[GazetteRecord] = []
    filtered: List[GazetteRecord] = []
    for rec in records:
        if rec.source_type.lower() == "act":
            rec.contract_domain = rec.contract_domain or ["contract"]
            kept.append(rec)
            continue
        excl = STATE_EXCLUDE.get(rec.state or "", [])
        basis_raw = " | ".join(filter(None, [rec.title or "", rec.notification_number or ""]))
        lower_basis = basis_raw.lower()
        if any(term in lower_basis for term in excl):
            filtered.append(rec)
            continue
        basis_parts = [rec.title or "", rec.notification_number or ""]
        basis = " | ".join(p for p in basis_parts if p)
        domains = _tag_domains(basis)
        if not domains and rec.state in STATE_INCLUDE:
            include_map = STATE_INCLUDE[rec.state]
            for domain_name, keywords in include_map.items():
                if any(kw in lower_basis for kw in keywords):
                    domains.append(domain_name)
        if domains:
            rec.contract_domain = domains
            kept.append(rec)
        else:
            filtered.append(rec)
    return kept, filtered


def normalize_record(
    *,
    source_type: str,
    jurisdiction: str,
    state: str | None,
    gazette_type: str | None,
    title: str | None,
    pdf_url: str,
    pdf_hash_value: str,
    source_page_url: str,
    pdf_bytes: bytes,
    listing_title: str | None = None,
    listing_notification: str | None = None,
    listing_date: str | None = None,
) -> GazetteRecord:
    pdf_text = extract_pdf_text(pdf_bytes)
    chosen_title = (title or listing_title or pdf_url.split("/")[-1]).strip()
    if not chosen_title:
        raise ValueError("missing title after normalization")

    notif = extract_notification_number(pdf_text, listing_hint=listing_notification)

    # Prefer explicit listing date when present; fall back to text-derived dates.
    listing_normalized = _normalize_date_str(listing_date) if listing_date else None
    effective_date = extract_effective_date(pdf_text, listing_date=None) or listing_normalized

    if not effective_date:
        effective_date = listing_date

    if not effective_date:
        raise ValueError("missing effective_date after normalization")
    if not notif:
        fallback_state = state or jurisdiction or "UNKNOWN"
        fallback_gazette = gazette_type or "NA"
        notif = f"{fallback_state}|{fallback_gazette}|{effective_date}|{pdf_hash_value[:8]}"

    rec = GazetteRecord(
        source_type=source_type,
        jurisdiction=jurisdiction,
        state=state,
        gazette_type=gazette_type,
        title=chosen_title,
        notification_number=notif,
        date=effective_date,
        pdf_url=pdf_url,
        pdf_hash=pdf_hash_value,
        source_page_url=source_page_url,
    )
    return rec
