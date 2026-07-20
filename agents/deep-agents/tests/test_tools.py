from pathlib import Path

from tools.pii_redact import redact_pii
from tools.pdf_extract import extract_pdf_text
from tools.vector_store import store_documents
from tools.rag_query import query_store


def test_redact_pii_masks_email_and_account():
    raw = "Email jane.doe@example.com account 12345678 phone +1-555-0100"
    out = redact_pii(raw)
    assert "jane.doe@example.com" not in out
    assert "12345678" not in out
    assert "[REDACTED" in out or "***" in out


def test_extract_pdf_text_reads_sample(tmp_path: Path):
    sample = Path(__file__).resolve().parents[3] / "data" / "bank-statement-document" / "Dummy-Bank-Statement.pdf"
    if not sample.exists():
        # minimal synthetic pdf via pymupdf
        import fitz
        sample = tmp_path / "mini.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Opening Balance 100\nCoffee 5.00 Balance 95")
        doc.save(sample)
        doc.close()
    text = extract_pdf_text(sample)
    assert isinstance(text, str)
    assert len(text.strip()) > 0


def test_store_and_query_roundtrip(tmp_path: Path):
    persist = tmp_path / "vs"
    store_documents(
        ["Total debits were 250.00 dollars for groceries and rent."],
        persist_dir=persist,
    )
    answer = query_store("What were total debits?", persist_dir=persist, k=2)
    assert isinstance(answer, str)
    assert len(answer) > 0
