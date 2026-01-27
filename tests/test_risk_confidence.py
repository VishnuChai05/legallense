import json
import backend.app as backend

class DummyDoc:
    def __init__(self, content: str):
        self.page_content = content

class DummyVectorStore:
    def __init__(self, texts):
        self.texts = texts
    def similarity_search(self, query, k=2):
        # return first k docs regardless of query
        return [DummyDoc(t) for t in self.texts[:k]]


def test_normalize_risk_payload_derives_confidence_when_missing():
    # No explicit confidence from model -> derive from evidence and coverage
    raw = json.dumps({
        "score": 42,
        "top_risks": [
            {"issue": "Liability cap missing", "evidence": "no cap clause", "severity": "High"},
            {"issue": "Payment unclear", "severity": "Medium"},
        ],
        "rationale": "Key caps and payment terms unclear"
    })
    payload = backend._normalize_risk_payload(raw, snippet_count=6)
    assert payload["score"] == 42
    assert payload["level"] == "Medium"
    assert payload["confidence"] > 0.3
    assert payload["confidence_level"] in {"Low", "Medium", "High"}


def test_normalize_risk_payload_uses_model_confidence_if_present():
    raw = json.dumps({
        "score": 70,
        "level": "High",
        "confidence": 0.9,
        "top_risks": [
            {"issue": "Uncapped indemnity", "evidence": "indemnity without cap"}
        ]
    })
    payload = backend._normalize_risk_payload(raw, snippet_count=2)
    assert payload["confidence"] == 0.9
    assert payload["confidence_level"] == "High"


def test_execute_risk_calculation_includes_confidence(monkeypatch):
    # Force deterministic docs and LLM output
    backend.RISK_TOPICS[:] = ["topic1"]
    vector_store = DummyVectorStore(["Clause A", "Clause B", "Clause C"])

    def fake_llm_call(prompt: str):
        class R:
            content = json.dumps({
                "score": 55,
                "level": "Medium",
                "top_risks": [{"issue": "Test risk", "evidence": "Clause A"}],
            })
        return R()

    monkeypatch.setattr(backend, "_get_risk_llm", lambda: type("LLM", (), {"invoke": lambda self, p: fake_llm_call(p)})())

    result = backend._execute_risk_calculation(vector_store)
    assert 0 <= result["confidence"] <= 1
    assert result["confidence_level"] in {"Low", "Medium", "High"}
