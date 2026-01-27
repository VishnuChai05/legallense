import asyncio
import json
import os
import time
from dataclasses import asdict
from typing import Awaitable, Callable, Dict, List, Tuple

from .exporter import export_jsonl
from .gazette_model import GazetteRecord
from .normalizer import apply_relevance, dedupe_records
from .sources_playwright import run_delhi, run_india_code_transfer_property, run_karnataka

ScraperFn = Callable[[], Awaitable[List[GazetteRecord]]]

SCRAPERS: Dict[str, ScraperFn] = {
    "india_code_transfer_property": run_india_code_transfer_property,
    "karnataka_gazette": run_karnataka,
    "delhi_gazette": run_delhi,
}

DEFAULT_TIMEOUT_SECONDS = int(os.getenv("SCRAPE_TIMEOUT_SECONDS", "60"))


async def _run_scraper(name: str, fn: ScraperFn, timeout_s: int) -> Tuple[str, List[GazetteRecord], str | None]:
    try:
        result = await asyncio.wait_for(fn(), timeout=timeout_s)
        return name, result, None
    except Exception as exc:  # pragma: no cover - defensive
        return name, [], str(exc)


async def run_all(timeout_s: int = DEFAULT_TIMEOUT_SECONDS) -> Dict[str, object]:
    start = time.time()
    results: List[GazetteRecord] = []
    summary: Dict[str, Dict[str, object]] = {}

    for name, fn in SCRAPERS.items():
        scraper_start = time.time()
        scr_name, recs, err = await _run_scraper(name, fn, timeout_s)
        duration = round(time.time() - scraper_start, 2)
        if err:
            summary[scr_name] = {"status": "error", "error": err, "count": 0, "seconds": duration}
        else:
            summary[scr_name] = {"status": "ok", "count": len(recs), "seconds": duration}
            results.extend(recs)

    relevant, filtered = apply_relevance(results)
    deduped = dedupe_records(relevant)
    payload = [asdict(r) for r in deduped]
    out_path = export_jsonl(payload, filename="scrape_results_playwright.jsonl")
    if filtered:
        export_jsonl([asdict(r) for r in filtered], filename="scrape_results_playwright_filtered.jsonl")

    end = time.time()
    manifest = {
        "total_records": len(results),
        "relevant_records": len(relevant),
        "filtered_no_domain": len(filtered),
        "deduped_records": len(deduped),
        "duration_seconds": round(end - start, 2),
        "summary": summary,
        "export_path": str(out_path),
    }
    return manifest


def main():
    manifest = asyncio.run(run_all())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
