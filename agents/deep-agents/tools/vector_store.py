from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class HashEmbedding(Embeddings):
    """Offline deterministic embedding for tests/local smoke without API keys."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for i, ch in enumerate(text.encode("utf-8")):
            vec[i % self.dim] += (ch % 31) / 31.0
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def store_documents(texts: list[str], persist_dir: str | Path) -> str:
    path = Path(persist_dir)
    path.mkdir(parents=True, exist_ok=True)
    docs = [Document(page_content=t) for t in texts if t and t.strip()]
    if not docs:
        raise ValueError("no documents to store")
    Chroma.from_documents(
        documents=docs,
        embedding=HashEmbedding(),
        persist_directory=str(path),
    )
    return str(path.resolve())
