from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import TOP_K


def retrieve_documents(
    vector_store: Chroma,
    query: str,
) -> list[Document]:
    """
    Retrieve the most semantically relevant chunks
    for a user question.
    """

    query = query.strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    documents = (
        vector_store.similarity_search(
            query=query,
            k=TOP_K,
        )
    )

    return documents