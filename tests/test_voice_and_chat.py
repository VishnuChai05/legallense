import types
from types import SimpleNamespace
import pytest

import app as app_module


class FakeDoc:
    def __init__(self, text):
        self.page_content = text


class FakeVectorStore:
    def __init__(self, texts):
        self.texts = texts

    def similarity_search(self, question, k=5):
        return [FakeDoc(t) for t in self.texts[:k]]


def test_execute_rag_query_uses_llm(monkeypatch):
    fake_store = FakeVectorStore(["Clause 1: Termination."])

    class FakeLLM:
        def invoke(self, prompt):
            return SimpleNamespace(content="RAG answer based on context")

    monkeypatch.setattr(app_module, "_get_chat_llm", lambda: FakeLLM())

    result = app_module._execute_rag_query(fake_store, "What is termination?")
    assert "RAG answer based on context" in result


def test_audio_response_openai_success(monkeypatch):
    fake_store = FakeVectorStore(["Payment terms: 30 days."])

    monkeypatch.setattr(app_module, "_transcribe_with_openai", lambda b: ("What is payment term?", "whisper-1"))
    monkeypatch.setattr(app_module, "get_rag_response", lambda vs, q=None: "Payment due in 30 days.")

    resp = app_module.get_audio_response(b"dummy-audio-bytes", fake_store)
    assert resp.get("question") is not None
    assert "Payment due in 30 days." == resp.get("answer")


def test_audio_response_openai_quota_then_google_success(monkeypatch):
    fake_store = FakeVectorStore(["Liability: limited."])

    def raise_quota(_):
        raise Exception("429 Too Many Requests - quota exceeded")

    monkeypatch.setattr(app_module, "_transcribe_with_openai", raise_quota)
    monkeypatch.setattr(app_module, "_transcribe_with_google_free", lambda b: ("What about liability?", "google_free"))
    monkeypatch.setattr(app_module, "get_rag_response", lambda vs, q=None: "Liability is limited.")

    resp = app_module.get_audio_response(b"dummy-audio-bytes", fake_store)
    assert resp.get("answer") == "Liability is limited."
    assert "transcribed via free Google" in resp.get("question") or "google_free" in resp.get("question")


def test_audio_response_openai_non_quota_failure(monkeypatch):
    fake_store = FakeVectorStore(["Text."])

    def raise_err(_):
        raise Exception("Some other error")

    monkeypatch.setattr(app_module, "_transcribe_with_openai", raise_err)

    resp = app_module.get_audio_response(b"dummy-audio-bytes", fake_store)
    assert resp.get("error") is not None
    assert "Transcription error" in resp.get("error")
