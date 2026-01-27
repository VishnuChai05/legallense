from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class Source:
    name: str
    url: str
    kind: str  # "html" or "pdf"
    topic: str  # e.g., "contract", "rental", "employment"
    jurisdiction: str  # e.g., "india", "state-tamil-nadu"
    requires_browser: bool = False
    notes: str = ""


def default_sources() -> List[Source]:
    """Seed sources focused on Indian contract/rental/employment laws.

    NOTE: Always review robots.txt and site terms before scraping.
    The list is a starting point; expand per compliance.
    """
    return [
        Source(
            name="India Code - Contract Act 1872",
            url="https://www.indiacode.nic.in/handle/123456789/2187?sam_handle=123456789/1362",
            kind="pdf",
            topic="contract",
            jurisdiction="india",
            notes="Primary contract legislation; PDF available via IndiaCode.",
        ),
        Source(
            name="India Code - Specific Relief Act",
            url="https://www.indiacode.nic.in/handle/123456789/2183",
            kind="pdf",
            topic="contract",
            jurisdiction="india",
            notes="Specific performance and reliefs relevant to agreements.",
        ),
        Source(
            name="eGazette (Central) - Weekly",
            url="https://legislative.gov.in/actsofparliamentfromtheyear/1950",
            kind="html",
            topic="notifications",
            jurisdiction="india",
            notes="Acts of Parliament list; stable HTTPS endpoint.",
        ),
        Source(
            name="Maharashtra Govt Acts - Rent",
            url="https://mls.org.in/#/home",
            kind="html",
            topic="rental",
            jurisdiction="state-maharashtra",
            notes="Legislative portal; may require navigation and session cookies.",
        ),
        Source(
            name="Tamil Nadu Labour Department - Notifications",
            url="https://labour.tn.gov.in/",
            kind="html",
            topic="employment",
            jurisdiction="state-tamil-nadu",
            notes="Landing page; notifications reachable via navigation.",
        ),
    ]


def load_sources_from_yaml(path: str | Path) -> List[Source]:
    data = yaml.safe_load(Path(path).read_text()) or []
    if not isinstance(data, list):
        raise ValueError("sources.yaml must be a list of source mappings")

    sources: List[Source] = []
    for item in data:
        sources.append(
            Source(
                name=item.get("name", ""),
                url=item.get("url", ""),
                kind=item.get("kind", "html"),
                topic=item.get("topic", "unknown"),
                jurisdiction=item.get("jurisdiction", "unknown"),
                requires_browser=bool(item.get("requires_browser", False)),
                notes=item.get("notes", ""),
            )
        )
    return sources
