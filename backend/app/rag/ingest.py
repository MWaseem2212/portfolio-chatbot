from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.rag.loaders import load_all
from app.rag.retriever import get_embeddings

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def ingest() -> int:
    settings = get_settings()
    chunks = split_documents(load_all())
    QdrantVectorStore.from_documents(
        chunks,
        embedding=get_embeddings(),
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value() or None,
        collection_name=settings.qdrant_collection,
        force_recreate=True,
    )
    return len(chunks)


if __name__ == "__main__":
    print(f"{ingest()} chunks Qdrant mein store ho gaye")