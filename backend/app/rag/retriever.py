from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)