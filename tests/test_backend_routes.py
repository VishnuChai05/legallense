import os
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import backend.app as backend
import time


class DummyVectorStore:
    def __init__(self, texts=None):
        self.texts = texts or ["sample"]

    def similarity_search(self, question, k=5):
        class Doc:
            def __init__(self, content):
                self.page_content = content

        return [Doc(t) for t in self.texts[:k]]


def client():
    return TestClient(backend.app)


def test_health_ok():
    resp = client().get("/health", headers={"x-user-id": "u1"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_upload_contract_happy_path(monkeypatch):
    def fake_validate(uploaded_file):
        return b"%PDF-1.4\n..."

    def fake_build(user_id, file_hash, file_bytes):
        now = time.time()
        key = f"{user_id}:{file_hash}"
        backend.vector_store_cache[key] = (DummyVectorStore(), now)
        backend.contract_bytes_cache[key] = (file_bytes, now)
        return {"file_hash": file_hash, "embedding_source": "test"}

    monkeypatch.setattr(backend, "_validate_pdf_upload", fake_validate)
    monkeypatch.setattr(backend, "_build_vector_store", fake_build)

    files = {"file": ("doc.pdf", BytesIO(b"hello"), "application/pdf")}
    resp = client().post("/contracts/upload", files=files, headers={"x-user-id": "u1"})

    assert resp.status_code == 200
    data = resp.json()
    assert "file_hash" in data
    assert data["meta"]["embedding_source"] == "test"
    assert f"u1:{data['file_hash']}" in backend.vector_store_cache


def test_chat_requires_vector_store():
    backend.vector_store_cache.clear()
    resp = client().post("/chat", json={"file_hash": "missing", "question": "Hi"}, headers={"x-user-id": "u1"})
    assert resp.status_code == 404


def test_chat_rejects_long_question():
    backend.vector_store_cache.clear()
    backend.vector_store_cache["u1:fh"] = (DummyVectorStore(["Answer text"]), time.time())
    long_q = "x" * (backend.MAX_QUESTION_CHARS + 10)
    resp = client().post("/chat", json={"file_hash": "fh", "question": long_q}, headers={"x-user-id": "u1"})
    assert resp.status_code == 400


def test_rate_limit_blocks(monkeypatch):
    backend.rate_limit_state.clear()
    backend.fallback_rate_state.clear()
    monkeypatch.setattr(backend, "RATE_LIMIT_PER_MIN", 1)
    monkeypatch.setattr(backend, "RATE_LIMIT_BURST", 0)

    c = client()
    assert c.get("/health", headers={"x-user-id": "u1"}).status_code == 200
    resp = c.get("/health", headers={"x-user-id": "u1"})
    assert resp.status_code == 429


def test_chat_returns_answer(monkeypatch):
    backend.vector_store_cache.clear()
    backend.vector_store_cache["u1:fh"] = (DummyVectorStore(["Answer text"]), time.time())
    monkeypatch.setattr(backend, "_execute_rag_query", lambda vs, q: "stubbed answer")

    resp = client().post("/chat", json={"file_hash": "fh", "question": "?"}, headers={"x-user-id": "u1"})

    assert resp.status_code == 200
    assert resp.json()["answer"] == "stubbed answer"


def test_risk_returns_payload(monkeypatch):
    backend.vector_store_cache.clear()
    backend.vector_store_cache["u1:fh"] = (DummyVectorStore(["Risk text"]), time.time())
    monkeypatch.setattr(
        backend,
        "_execute_risk_calculation",
        lambda vs: {"score": 10, "level": "Low", "top_risks": ["None"]},
    )

    resp = client().post("/risk", json={"file_hash": "fh"}, headers={"x-user-id": "u1"})

    assert resp.status_code == 200
    assert resp.json()["score"] == 10