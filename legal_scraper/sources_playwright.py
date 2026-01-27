import asyncio
import json
import re
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from .gazette_model import GazetteRecord, hash_bytes
from .normalizer import normalize_record
from .playwright_utils import browser_context, collect_pdf_responses, fetch_pdf_bytes, wait_for_network_idle


async def run_india_code_transfer_property() -> List[GazetteRecord]:
    results: List[GazetteRecord] = []
    bitstream_url = "https://www.indiacode.nic.in/bitstream/123456789/2263/1/A1882-04.pdf"
    fallback_html_url = "https://www.advocatekhoj.com/library/bareActs/transferofproperty/index.php?Title=Transfer%20of%20Property%20Act,%201882"
    local_pdf_path = Path("data/raw/A1882-04.pdf")
    async with browser_context() as (browser, _close):
        page = await browser.new_page()
        pdf_responses = await collect_pdf_responses(page)
        await page.goto("https://www.indiacode.nic.in/handle/123456789/2263", wait_until="networkidle")
        await wait_for_network_idle(page)
        try:
            await page.click("a[href*='bitstream']", timeout=4000)
            await wait_for_network_idle(page)
        except Exception:
            pass
        # Primary: capture PDF response via network listeners
        for resp in pdf_responses:
            url, content = await fetch_pdf_bytes(resp)
            if not content:
                continue
            pdf_hash_value = hash_bytes(content)
            rec = normalize_record(
                source_type="Act",
                jurisdiction="India",
                state=None,
                gazette_type=None,
                title="Transfer of Property Act 1882",
                pdf_url=url,
                pdf_hash_value=pdf_hash_value,
                source_page_url=page.url,
                pdf_bytes=content,
                listing_date="1882-07-01",
                listing_notification="Transfer of Property Act 1882",
            )
            results.append(rec)
            break

        # Fallback: fetch the known bitstream directly if no PDF response was captured
        if not results:
            # Try fetching bitstream with page cookies (session-aware)
            resp = await page.request.get(bitstream_url, headers={"Referer": page.url})
            if resp.ok:
                content = await resp.body()
                if not content:
                    raise RuntimeError("IndiaCode bitstream fetch returned empty body")
                pdf_hash_value = hash_bytes(content)
                rec = normalize_record(
                    source_type="Act",
                    jurisdiction="India",
                    state=None,
                    gazette_type=None,
                    title="Transfer of Property Act 1882",
                    pdf_url=bitstream_url,
                    pdf_hash_value=pdf_hash_value,
                    source_page_url=page.url,
                    pdf_bytes=content,
                    listing_date="1882-07-01",
                    listing_notification="Transfer of Property Act 1882",
                )
                results.append(rec)
            elif local_pdf_path.exists():
                content = local_pdf_path.read_bytes()
                pdf_hash_value = hash_bytes(content)
                rec = normalize_record(
                    source_type="Act",
                    jurisdiction="India",
                    state=None,
                    gazette_type=None,
                    title="Transfer of Property Act 1882",
                    pdf_url=str(local_pdf_path),
                    pdf_hash_value=pdf_hash_value,
                    source_page_url=str(local_pdf_path),
                    pdf_bytes=content,
                    listing_date="1882-07-01",
                    listing_notification="Transfer of Property Act 1882",
                )
                results.append(rec)
            else:
                # Attempt external fallback source with PDF link extraction
                html_resp = await page.request.get(fallback_html_url)
                if not html_resp.ok:
                    raise RuntimeError(f"IndiaCode bitstream fetch failed: {resp.status} {resp.status_text}; fallback fetch failed: {html_resp.status} {html_resp.status_text}")
                html_content = await html_resp.text()
                pdf_match = re.search(r"href=\"([^\"]+\.pdf)\"", html_content, re.IGNORECASE)
                if not pdf_match:
                    raise RuntimeError("Fallback source did not contain a PDF link and no local PDF present")
                pdf_href = pdf_match.group(1)
                pdf_url = urljoin(fallback_html_url, pdf_href)
                pdf_resp = await page.request.get(pdf_url)
                if not pdf_resp.ok:
                    raise RuntimeError(f"Fallback PDF fetch failed: {pdf_resp.status} {pdf_resp.status_text}")
                content = await pdf_resp.body()
                if not content:
                    raise RuntimeError("Fallback PDF returned empty body")
                pdf_hash_value = hash_bytes(content)
                rec = normalize_record(
                    source_type="Act",
                    jurisdiction="India",
                    state=None,
                    gazette_type=None,
                    title="Transfer of Property Act 1882",
                    pdf_url=pdf_url,
                    pdf_hash_value=pdf_hash_value,
                    source_page_url=fallback_html_url,
                    pdf_bytes=content,
                    listing_date="1882-07-01",
                    listing_notification="Transfer of Property Act 1882",
                )
                results.append(rec)
    return results


def _extract_listing_metadata(rows_text: List[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Best-effort extraction of title/notification/date from listing rows before click."""
    date_re = re.compile(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}")
    notification_re = re.compile(r"\b(?:G\.S\.R\.|S\.O\.|No\.)[^,\s]{0,40}", re.IGNORECASE)

    listing_title = None
    listing_notification = None
    listing_date = None

    for row_text in rows_text:
        if not listing_title:
            listing_title = row_text.strip() or None
        if not listing_notification:
            m = notification_re.search(row_text)
            if m:
                listing_notification = m.group(0).strip()
        if not listing_date:
            m = date_re.search(row_text)
            if m:
                listing_date = m.group(0).strip()
        if listing_title and listing_notification and listing_date:
            break
    return listing_title, listing_notification, listing_date


async def _scrape_gazette_listing(url: str, state_code: str, gazette_type: str) -> List[GazetteRecord]:
    records: List[GazetteRecord] = []
    async with browser_context() as (browser, _close):
        page = await browser.new_page()
        pdf_responses = await collect_pdf_responses(page)
        listing_title = None
        listing_notification = None
        listing_date = None
        try:
            await page.goto(url, wait_until="networkidle", timeout=15000)
            await wait_for_network_idle(page)
            # Try to gather listing text from table rows if present
            row_texts: List[str] = []
            rows = await page.query_selector_all("tr")
            for row in rows:
                text = (await row.inner_text()) or ""
                if text:
                    row_texts.append(text)
            listing_title, listing_notification, listing_date = _extract_listing_metadata(row_texts)

            links = await page.query_selector_all("a")
            for link in links:
                href = (await link.get_attribute("href")) or ""
                onclick = (await link.get_attribute("onclick")) or ""
                text = (await link.inner_text()) or ""
                lower = href.lower() + onclick.lower()
                if not listing_title and text:
                    listing_title = text.strip()
                if ".pdf" in lower or "bitstream" in lower or "pdf" in onclick.lower():
                    try:
                        await link.click(timeout=4000)
                        await wait_for_network_idle(page)
                        break
                    except Exception:
                        continue
                if "download" in lower or "view" in lower:
                    try:
                        await link.click(timeout=4000)
                        await wait_for_network_idle(page)
                        break
                    except Exception:
                        continue
        except PlaywrightTimeoutError:
            pass
        for resp in pdf_responses:
            try:
                url_pdf, content = await fetch_pdf_bytes(resp)
            except Exception:
                continue
            if not content:
                continue
            pdf_hash_value = hash_bytes(content)
            try:
                rec = normalize_record(
                    source_type="Gazette",
                    jurisdiction="State",
                    state=state_code,
                    gazette_type=gazette_type,
                    title=listing_title,
                    pdf_url=url_pdf,
                    pdf_hash_value=pdf_hash_value,
                    source_page_url=url,
                    pdf_bytes=content,
                    listing_title=listing_title,
                    listing_notification=listing_notification,
                    listing_date=listing_date,
                )
                records.append(rec)
                break
            except Exception:
                continue

        # Fallback: if no PDF captured, try direct fetch of first PDF-like href
        if not records:
            try:
                anchors = await page.query_selector_all("a")
                for a in anchors:
                    href = (await a.get_attribute("href")) or ""
                    if not href:
                        continue
                    if ".pdf" not in href.lower() and "bitstream" not in href.lower():
                        continue
                    pdf_url = urljoin(url, href)
                    resp = await page.request.get(pdf_url)
                    if not resp.ok:
                        continue
                    content = await resp.body()
                    if not content:
                        continue
                    pdf_hash_value = hash_bytes(content)
                    rec = normalize_record(
                        source_type="Gazette",
                        jurisdiction="State",
                        state=state_code,
                        gazette_type=gazette_type,
                        title=listing_title or href.split("/")[-1],
                        pdf_url=pdf_url,
                        pdf_hash_value=pdf_hash_value,
                        source_page_url=url,
                        pdf_bytes=content,
                        listing_title=listing_title,
                        listing_notification=listing_notification,
                        listing_date=listing_date,
                    )
                    records.append(rec)
                    break
            except Exception:
                pass
    return records


async def run_karnataka() -> List[GazetteRecord]:
    weekly = await _scrape_gazette_listing(
        "https://kla.kar.nic.in/council/gazette/gazette.htm",
        "KA",
        "Weekly",
    )
    extraordinary = []  # extend when extraordinary endpoint is available
    return weekly + extraordinary


async def run_delhi() -> List[GazetteRecord]:
    weekly = await _scrape_gazette_listing(
        "http://it.delhigovt.nic.in/pis/noc/egaz_search_result.asp",
        "DL",
        "Weekly",
    )
    extraordinary = []  # extend when extraordinary endpoint is available
    return weekly + extraordinary


async def run_all() -> List[GazetteRecord]:
    tasks = [
        run_india_code_transfer_property,
        run_karnataka,
        run_delhi,
    ]
    results: List[GazetteRecord] = []
    for task_fn in tasks:
        try:
            res = await task_fn()
            results.extend(res)
        except Exception:
            continue
    # Deduplicate by primary key then by hash
    seen = set()
    deduped: List[GazetteRecord] = []
    for r in results:
        pk = r.primary_key() or r.pdf_hash
        if pk in seen:
            continue
        seen.add(pk)
        deduped.append(r)
    return deduped


def main():
    out = asyncio.run(run_all())
    print(json.dumps([r.to_dict() for r in out], indent=2))


if __name__ == "__main__":
    main()
