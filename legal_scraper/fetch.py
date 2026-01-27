import hashlib
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from pypdf import PdfReader
from slugify import slugify

DEFAULT_UA = os.getenv(
    "SCRAPE_UA",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
)
DEFAULT_TIMEOUT = httpx.Timeout(15.0, read=30.0)
DATA_DIR = Path(os.getenv("SCRAPE_DATA_DIR", "data"))
DATA_RAW = DATA_DIR / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)
THROTTLE_SECONDS = float(os.getenv("SCRAPE_THROTTLE_SECONDS", "0.5"))
_LAST_REQUEST: dict[str, float] = {}
_ROBOTS_CACHE: dict[str, RobotFileParser] = {}


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _throttle_if_needed(url: str):
    if THROTTLE_SECONDS <= 0:
        return
    host = urlparse(url).netloc
    last = _LAST_REQUEST.get(host)
    if last is not None:
        sleep_for = THROTTLE_SECONDS - (time.time() - last)
        if sleep_for > 0:
            time.sleep(sleep_for)
    _LAST_REQUEST[host] = time.time()


def _robots_allowed(url: str, user_agent: str) -> bool:
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    robots_url = urljoin(base, "/robots.txt")

    if base in _ROBOTS_CACHE:
        rp = _ROBOTS_CACHE[base]
    else:
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            with httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True, headers={"User-Agent": user_agent}) as client:
                resp = client.get(robots_url)
                if resp.status_code >= 400:
                    rp.allow_all = True
                else:
                    rp.parse(resp.text.splitlines())
        except Exception:
            rp.allow_all = True
        _ROBOTS_CACHE[base] = rp

    return rp.can_fetch(user_agent, url)


def fetch_bytes(url: str, headers: Optional[dict] = None, *, respect_robots: bool = False, throttle: bool = True) -> Tuple[bytes, str]:
    h = {"User-Agent": DEFAULT_UA, **(headers or {})}

    if respect_robots and not _robots_allowed(url, h["User-Agent"]):
        raise PermissionError(f"Blocked by robots.txt: {url}")

    if throttle:
        _throttle_if_needed(url)

    with httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True, headers=h) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content, resp.headers.get("content-type", "")


def fetch_html_browser(url: str, *, timeout: float = 30.0) -> Tuple[bytes, str]:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument(f"--user-agent={DEFAULT_UA}")
    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(timeout)
    try:
        driver.get(url)
        html = driver.page_source
        return html.encode("utf-8", errors="ignore"), "text/html"
    finally:
        driver.quit()


def parse_pdf(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts)


def parse_html(content: bytes) -> str:
    soup = BeautifulSoup(content, "lxml")
    # Drop script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" \n", strip=True)
    return text


def _find_pdf_link(html: bytes, base_url: str) -> Optional[str]:
    """Attempt to locate a PDF link inside an HTML page (anchors or iframes)."""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return None

    # Check anchors first
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        lower = href.lower()
        if ".pdf" in lower or "bitstream/" in lower or "filename=" in lower:
            return urljoin(base_url, href)

    # Check iframes (some gazette sites embed PDFs in iframe src)
    for tag in soup.find_all("iframe", src=True):
        src = tag["src"].strip()
        lower = src.lower()
        if ".pdf" in lower or "bitstream/" in lower or "filename=" in lower:
            return urljoin(base_url, src)

    return None


def save_raw(name: str, url: str, content: bytes, content_type: str) -> Path:
    slug = slugify(name) or _hash_bytes(content)
    ext = "pdf" if "pdf" in content_type.lower() else "html"
    out_path = DATA_RAW / f"{slug}.{ext}"
    meta_path = DATA_RAW / f"{slug}.meta"

    out_path.write_bytes(content)
    meta_path.write_text(f"url: {url}\ncontent_type: {content_type}\nsha256: {_hash_bytes(content)}\n")
    return out_path


def scrape_source(
    name: str,
    url: str,
    kind: str,
    *,
    respect_robots: bool = False,
    throttle: bool = True,
    requires_browser: bool = False,
) -> Tuple[str, Path]:
    """Fetch and extract text from a single source.

    Returns (text, raw_path).
    """
    fetched_url = url
    if requires_browser:
        try:
            content, ctype = fetch_html_browser(url)
        except Exception:
            # Fallback to standard fetch if browser path fails
            content, ctype = fetch_bytes(url, respect_robots=respect_robots, throttle=throttle)
    else:
        content, ctype = fetch_bytes(url, respect_robots=respect_robots, throttle=throttle)

    # If we expected a PDF but received HTML, try to follow the first PDF link we can find.
    if kind == "pdf" and "pdf" not in (ctype or "").lower():
        pdf_url = _find_pdf_link(content, url)
        if pdf_url:
            fetched_url = pdf_url
            content, ctype = fetch_bytes(pdf_url, respect_robots=respect_robots, throttle=throttle)

    # For HTML sources that contain a PDF link, follow it to ingest the document instead of the index page.
    if kind == "html" and "pdf" not in (ctype or "").lower():
        pdf_url = _find_pdf_link(content, url)
        if pdf_url:
            try:
                fetched_url = pdf_url
                content, ctype = fetch_bytes(pdf_url, respect_robots=respect_robots, throttle=throttle)
                kind = "pdf"
            except Exception:
                # If follow fails, keep original HTML content
                pass

    raw_path = save_raw(name, fetched_url, content, ctype or kind)

    if kind == "pdf" or "pdf" in (ctype or "").lower():
        try:
            text = parse_pdf(content)
        except Exception:
            # Fall back to HTML parsing if the payload is not a valid PDF
            text = parse_html(content)
    else:
        text = parse_html(content)
    return text, raw_path
