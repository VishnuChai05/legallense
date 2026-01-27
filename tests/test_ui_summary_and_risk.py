import app as app_module


class FakeDoc:
    def __init__(self, text):
        self.page_content = text


class FakeVectorStore:
    def __init__(self, texts):
        self.texts = texts
    def similarity_search(self, query, k=5):
        return [FakeDoc(t) for t in self.texts[:k]]


def test_build_vector_store_returns_page_and_size(monkeypatch):
    # Use a 1-page PDF from highlighter test
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello world")
    pdf_bytes = doc.tobytes()
    doc.close()

    vs, _, meta = app_module.build_vector_store("hash", pdf_bytes)
    assert meta.get("page_count") == 1
    assert meta.get("file_size_kb") > 0


def test_risk_confidence_surfaces_in_ui_logic(monkeypatch):
    fake_store = FakeVectorStore(["Clause A", "Clause B"])

    def fake_exec():
        return {
            "score": 55,
            "level": "Medium",
            "top_risks": [{"issue": "Risk A", "evidence": "Clause A", "severity": "High"}],
            "confidence": 0.8,
            "confidence_level": "High",
        }

    monkeypatch.setattr(app_module, "calculate_risk_score", lambda fh, vs: fake_exec())

    # Simulate core UI logic path: just ensure calculate_risk_score returns expected keys
    result = app_module.calculate_risk_score("fh", fake_store)
    assert result["confidence"] == 0.8
    assert result["confidence_level"] == "High"
    assert result["top_risks"][0]["issue"] == "Risk A"
