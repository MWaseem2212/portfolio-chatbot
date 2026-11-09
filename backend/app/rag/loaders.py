from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document

from app.config import get_settings


def load_resume() -> list[Document]:
    return PyPDFLoader(get_settings().resume_url).load()


def load_website() -> list[Document]:
    return WebBaseLoader(
        web_paths=[get_settings().site_url],
        bs_get_text_kwargs={"separator": "\n", "strip": True},
    ).load()


def load_all() -> list[Document]:
    return load_resume() + load_website()


if __name__ == "__main__":
    for name, loader in (("resume", load_resume), ("website", load_website)):
        docs = loader()
        print(f"\n===== {name}: {len(docs)} documents =====")
        for doc in docs:
            print(doc.metadata)
            print(doc.page_content[:500])
            print("...")