from functools import lru_cache

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_qdrant import QdrantVectorStore

from app.config import get_settings

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 3


def get_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache
def get_retriever() -> VectorStoreRetriever:
    settings = get_settings()
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        collection_name=settings.qdrant_collection,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value() or None,
    )
    return vector_store.as_retriever(search_kwargs={"k": TOP_K})