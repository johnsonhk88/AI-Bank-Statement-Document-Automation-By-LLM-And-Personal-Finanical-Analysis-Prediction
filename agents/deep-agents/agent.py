from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend

from tools.pdf_extract import extract_pdf_text
from tools.pii_redact import redact_pii
from tools.vector_store import store_documents
from tools.rag_query import query_store

ROOT = Path(__file__).resolve().parent
SKILLS = str(ROOT / "skills")
WORKSPACE = ROOT / "workspace"


def pdf_extract(path: str) -> str:
    """Extract text from a bank-statement PDF path."""
    return extract_pdf_text(path)


def pii_redact(text: str) -> str:
    """Redact PII from text before storage or display."""
    return redact_pii(text)


def vector_store(text: str, collection_name: str = "statements") -> str:
    """Store redacted text into the local vector DB; returns persist path."""
    persist = WORKSPACE / "vector_stores" / collection_name
    return store_documents([text], persist_dir=persist)


def rag_query(question: str, collection_name: str = "statements") -> str:
    """Answer a question using RAG over stored statement chunks."""
    persist = WORKSPACE / "vector_stores" / collection_name
    return query_store(question, persist_dir=persist)


def build_agent(model: str = "ollama:llama3.2"):
    """Create a deep agent with bank-statement tools and skills.

    Requires a configured LLM provider. For offline smoke tests use
    `run_pipeline` instead.
    """
    backend = FilesystemBackend(root_dir=str(ROOT))
    return create_deep_agent(
        model=model,
        tools=[pdf_extract, pii_redact, vector_store, rag_query],
        skills=[SKILLS],
        backend=backend,
        system_prompt=(
            "You are a bank-statement automation assistant. "
            "Follow loaded skills. Always extract PDF, redact PII before vector_store, "
            "then answer with rag_query. Prefer structured markdown output."
        ),
    )


def run_pipeline(pdf_path: str | Path, question: str, collection_name: str = "statements") -> str:
    """Deterministic E2E path without an LLM (tools only)."""
    raw = extract_pdf_text(pdf_path)
    clean = redact_pii(raw)
    store_documents([clean], persist_dir=WORKSPACE / "vector_stores" / collection_name)
    retrieval = query_store(question, persist_dir=WORKSPACE / "vector_stores" / collection_name)
    return (
        "# Bank Statement E2E Result\n\n"
        f"## Question\n{question}\n\n"
        f"## RAG retrieval\n{retrieval}\n\n"
        "## Notes\nPII redaction applied before vector store.\n"
    )
