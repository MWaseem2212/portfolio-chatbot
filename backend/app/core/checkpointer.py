from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver

from app.config import get_settings


@contextmanager
def postgres_checkpointer() -> Iterator[PostgresSaver]:
    url = get_settings().postgres_url
    if not url:
        raise RuntimeError("POSTGRES_URL .env mein set nahi hai")
    with PostgresSaver.from_conn_string(url) as checkpointer:
        checkpointer.setup()
        yield checkpointer