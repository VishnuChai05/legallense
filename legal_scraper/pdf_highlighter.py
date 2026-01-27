"""PDF highlighter utility.

Provides a small helper to highlight matched terms in a PDF and write an annotated copy.
Uses PyMuPDF (fitz) under the hood.
"""

from pathlib import Path
from typing import Iterable, Sequence

import fitz  # PyMuPDF


def highlight_terms_in_pdf(
    pdf_path: str | Path,
    terms: Sequence[str],
    output_path: str | Path | None = None,
) -> Path:
    """Highlight all occurrences of the given terms in red and save a copy.

    Args:
        pdf_path: Input PDF file path.
        terms: List of case-insensitive terms/phrases to highlight.
        output_path: Where to write the annotated PDF. If None, suffix `_annotated` is used.

    Returns:
        Path to the annotated PDF.
    """

    in_path = Path(pdf_path)
    if not in_path.exists():
        raise FileNotFoundError(f"PDF not found: {in_path}")

    out_path = Path(output_path) if output_path else in_path.with_stem(in_path.stem + "_annotated")

    doc = fitz.open(in_path)
    try:
        for page in doc:
            for term in terms:
                if not term:
                    continue
                hits = []
                # PyMuPDF search_for is case-sensitive; attempt a few variants
                for candidate in {term, term.lower(), term.upper(), term.title()}:
                    try:
                        hits.extend(page.search_for(candidate))
                    except Exception:
                        continue
                for rect in hits:
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=(1, 0, 0))  # red outline
                    annot.update()
        doc.save(out_path)
    finally:
        doc.close()

    return out_path
