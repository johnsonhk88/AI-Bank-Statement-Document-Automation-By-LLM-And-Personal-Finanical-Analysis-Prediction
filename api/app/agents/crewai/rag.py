from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings

_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    return _embeddings


def store_in_qdrant(text: str, collection_name: str) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = splitter.create_documents([text])
    QdrantVectorStore.from_documents(
        docs,
        embedding=_get_embeddings(),
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
        collection_name=collection_name,
    )


def query_qdrant(question: str, collection_name: str, k: int = 4) -> str:
    store = QdrantVectorStore.from_existing_collection(
        embedding=_get_embeddings(),
        collection_name=collection_name,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
    )
    results = store.similarity_search(question, k=k)
    return "\n\n".join([r.page_content for r in results])
