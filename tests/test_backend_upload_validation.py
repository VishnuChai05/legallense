import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile
from io import BytesIO

from backend.app import _validate_pdf_upload, MAX_FILE_SIZE_MB

SAMPLE_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 24 Tf 100 700 Td (Hello) Tj ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000010 00000 n 
0000000060 00000 n 
0000000115 00000 n 
0000000190 00000 n 
trailer
<< /Root 1 0 R /Size 5 >>
startxref
260
%%EOF"""


def make_upload(name: str, data: bytes, content_type: str = "application/pdf") -> UploadFile:
    # Starlette's UploadFile takes headers instead of a content_type kwarg.
    return UploadFile(filename=name, file=BytesIO(data), headers={"content-type": content_type})


def test_backend_accepts_valid_pdf():
    upload = make_upload("contract.pdf", SAMPLE_PDF)

    result = _validate_pdf_upload(upload)

    assert result.startswith(b"%PDF")


def test_backend_rejects_non_pdf_extension_even_octet_stream():
    upload = make_upload("contract.txt", b"data", content_type="application/octet-stream")

    with pytest.raises(HTTPException) as exc:
        _validate_pdf_upload(upload)

    assert exc.value.status_code == 400
    assert "Only PDF files" in exc.value.detail


def test_backend_rejects_empty_file():
    upload = make_upload("contract.pdf", b"")

    with pytest.raises(HTTPException) as exc:
        _validate_pdf_upload(upload)

    assert exc.value.status_code == 400
    assert "empty" in exc.value.detail.lower()


def test_backend_rejects_files_over_size_limit():
    over_limit = (MAX_FILE_SIZE_MB * 1024 * 1024) + 1
    upload = make_upload("contract.pdf", b"x" * over_limit)

    with pytest.raises(HTTPException) as exc:
        _validate_pdf_upload(upload)

    assert exc.value.status_code == 400
    assert "too large" in exc.value.detail.lower()
