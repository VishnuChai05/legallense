import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def _data_dir() -> Path:
    return Path(os.getenv("SCRAPE_DATA_DIR", "data"))


def _export_dir() -> Path:
    out = _data_dir() / "exports"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _vector_dir() -> Path:
    out = _data_dir() / "vector"
    out.mkdir(parents=True, exist_ok=True)
    return out


def export_jsonl(records: Iterable[Mapping[str, Any]], filename: str | None = None) -> Path:
    path = _export_dir() / (filename or "scrape_results.jsonl")
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True))
            f.write("\n")
    return path


_EMBED_MODEL: SentenceTransformer | None = None


def _get_embed_model(model_name: str) -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(model_name)
    return _EMBED_MODEL


def vector_ingest(
    records: Iterable[Mapping[str, Any]],
    *,
    model_name: str | None = None,
    index_name: str | None = None,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Build a FAISS index from scraped records.

    Requires each record to include a "text" field.
    """
    texts: list[str] = []
    metas: list[Mapping[str, Any]] = []
    for rec in records:
        text = rec.get("text")
        if not text:
            raise ValueError("All records must include 'text' for ingestion")
        texts.append(str(text))
        metas.append(rec)

    if not texts:
        return {"ingested": 0, "status": "empty"}

    embedder = embed_fn or _get_embed_model(model_name or os.getenv("SCRAPE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")).encode
    embeddings = embedder(texts)
    vectors = np.array(embeddings, dtype="float32")
    if vectors.ndim != 2:
        raise ValueError("Embedding function must return a 2D array")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    vector_dir = _vector_dir()
    base_name = index_name or "scrape_index"
    index_path = vector_dir / f"{base_name}.faiss"
    meta_path = vector_dir / f"{base_name}.jsonl"

    faiss.write_index(index, str(index_path))
    with meta_path.open("w", encoding="utf-8") as f:
        for rec in metas:
            f.write(json.dumps(rec, ensure_ascii=True))
            f.write("\n")

    return {
        "ingested": len(texts),
        "dim": vectors.shape[1],
        "index_path": str(index_path),
        "meta_path": str(meta_path),
        "status": "ok",
    }
