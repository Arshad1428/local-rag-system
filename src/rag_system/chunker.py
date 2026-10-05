from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Split loaded documents into overlapping chunks
    while preserving their metadata.
    """

    if not documents:
        raise ValueError(
            "No documents were provided for chunking."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        length_function=len,
    )

    chunks = splitter.split_documents(
        documents
    )

    chunks = [
        chunk
        for chunk in chunks
        if chunk.page_content.strip()
    ]

    if not chunks:
        raise ValueError(
            "Document chunking produced no text."
        )

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    return chunks