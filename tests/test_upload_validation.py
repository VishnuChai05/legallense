import pytest

from app import MAX_FILE_SIZE_MB, _validate_pdf_upload


class DummyUpload:
    def __init__(self, name: str, size: int, data: bytes):
        self.name = name
        self.size = size
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_accepts_valid_pdf_under_limit():
    data = (
        b"%PDF-1.1\n"
        b"1 0 obj<<>>endobj\n"
        b"2 0 obj<< /Type /Catalog /Pages 3 0 R >>endobj\n"
        b"3 0 obj<< /Type /Pages /Kids [4 0 R] /Count 1 >>endobj\n"
        b"4 0 obj<< /Type /Page /Parent 3 0 R /MediaBox [0 0 612 792] /Contents 5 0 R >>endobj\n"
        b"5 0 obj<< /Length 12 >>stream\nBT /F1 12 Tf ET\nendstream endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000120 00000 n \n0000000200 00000 n \n0000000300 00000 n \n"
        b"trailer<< /Root 2 0 R /Size 6 >>\nstartxref\n360\n%%EOF\n"
    )
    upload = DummyUpload("contract.pdf", len(data), data)

    result = _validate_pdf_upload(upload)

    assert result == data


def test_rejects_non_pdf_extension():
    upload = DummyUpload("contract.txt", 100, b"text")

    with pytest.raises(ValueError, match="Only PDF files are supported"):
        _validate_pdf_upload(upload)


def test_rejects_files_over_size_limit():
    over_limit = (MAX_FILE_SIZE_MB * 1024 * 1024) + 1
    upload = DummyUpload("contract.pdf", over_limit, b"x" * 10)

    with pytest.raises(ValueError, match="File too large"):
        _validate_pdf_upload(upload)
