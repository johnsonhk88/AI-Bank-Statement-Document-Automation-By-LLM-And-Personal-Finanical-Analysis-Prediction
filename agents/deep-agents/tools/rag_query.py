from pathlib import Path

from langchain_chroma import Chroma

from tools.vector_store import HashEmbedding


def query_store(question: str, persist_dir: str | Path, k: int = 4) -> str:
    path = Path(persist_dir)
    if not path.exists():
        raise FileNotFoundError(f"vector store not found: {path}")
    vs = Chroma(persist_directory=str(path), embedding_function=HashEmbedding())
    docs = vs.similarity_search(question, k=k)
    if not docs:
        return "No relevant documents found."
    chunks = "\n---\n".join(d.page_content for d in docs)
    return f"Question: {question}\n\nRetrieved context:\n{chunks}"
