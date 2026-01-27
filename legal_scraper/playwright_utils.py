import asyncio
import contextlib
import time
from typing import AsyncIterator, Callable, Dict, List, Optional, Tuple

from playwright.async_api import async_playwright, Browser, Page, Response

CRAWL_DELAY_SECONDS = 2.5
MAX_PARALLEL = 3


@contextlib.asynccontextmanager
async def browser_context() -> AsyncIterator[Tuple[Browser, Callable[[], asyncio.Task]]]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        async def close_browser():
            await browser.close()

        try:
            yield browser, close_browser
        finally:
            with contextlib.suppress(Exception):
                await browser.close()


async def wait_for_network_idle(page: Page, idle_ms: int = 1000, timeout_ms: int = 15000) -> None:
    start = time.time()
    last = time.time()

    def _on_request(_):
        nonlocal last
        last = time.time()

    def _on_response(_):
        nonlocal last
        last = time.time()

    page.on("request", _on_request)
    page.on("response", _on_response)

    try:
        while (time.time() - start) * 1000 < timeout_ms:
            await asyncio.sleep(0.1)
            if (time.time() - last) * 1000 >= idle_ms:
                return
    finally:
        with contextlib.suppress(Exception):
            page.remove_listener("request", _on_request)
            page.remove_listener("response", _on_response)


async def collect_pdf_responses(page: Page) -> List[Response]:
    pdf_responses: List[Response] = []

    def _capture(resp: Response):
        try:
            ctype = resp.headers.get("content-type", "").lower()
            if "pdf" in ctype:
                pdf_responses.append(resp)
        except Exception:
            pass

    page.on("response", _capture)
    return pdf_responses


async def fetch_pdf_bytes(response: Response) -> Tuple[str, bytes]:
    url = response.url
    body = await response.body()
    return url, body


async def throttled_gather(coros: List[Callable[[], asyncio.Task]]):
    semaphore = asyncio.Semaphore(MAX_PARALLEL)

    async def _runner(fn: Callable[[], asyncio.Task]):
        async with semaphore:
            await asyncio.sleep(CRAWL_DELAY_SECONDS)
            return await fn()

    return await asyncio.gather(*[_runner(c) for c in coros])
