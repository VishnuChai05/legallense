from dataclasses import dataclass, asdict
from hashlib import sha256
from typing import Optional, Dict, Any, List


@dataclass
class GazetteRecord:
    source_type: str  # "Act" | "Gazette"
    jurisdiction: str  # "India" | "State"
    state: Optional[str]  # e.g., "KA", "DL", etc.
    gazette_type: Optional[str]  # "Weekly" | "Extraordinary" | None
    title: str
    notification_number: Optional[str]
    date: str  # YYYY-MM-DD
    pdf_url: str
    pdf_hash: str
    source_page_url: str
    contract_domain: List[str] | None = None

    def primary_key(self) -> str:
        return "|".join(
            [
                self.source_type or "",
                self.state or "",
                self.gazette_type or "",
                self.notification_number or "",
                self.date or "",
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def hash_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()
