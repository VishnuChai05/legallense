from pathlib import Path

def test_export_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRAPE_DATA_DIR", str(tmp_path))
    import importlib
    import legal_scraper.exporter as exporter
    importlib.reload(exporter)

    records = [
        {"name": "Doc", "url": "http://example.com", "text": "hello"},
        {"name": "Doc2", "url": "http://example.org", "text": "world"},
    ]
    path = exporter.export_jsonl(records, filename="test.jsonl")
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_vector_ingest(tmp_path, monkeypatch):
    records = [
        {"name": "a", "text": "alpha"},
        {"name": "b", "text": "beta"},
    ]

    def fake_embed(texts):  # deterministic small vectors
        import numpy as np
        return np.stack([np.arange(3, dtype="float32") + i for i in range(len(texts))])

    import importlib
    monkeypatch.setenv("SCRAPE_DATA_DIR", str(tmp_path))
    import legal_scraper.exporter as exporter
    importlib.reload(exporter)

    summary = exporter.vector_ingest(records, index_name="test", embed_fn=fake_embed)
    assert summary["ingested"] == 2
    assert summary["dim"] == 3
    assert Path(summary["index_path"]).exists()
    assert Path(summary["meta_path"]).exists()
