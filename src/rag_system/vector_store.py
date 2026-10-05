import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
)
from .embeddings import (
    create_embedding_model,
)


def create_vector_store(
    chunks: list[Document],
) -> Chroma:
    """
    Build a fresh persistent ChromaDB collection
    from document chunks.
    """

    if not chunks:
        raise ValueError(
            "Cannot build a vector store without chunks."
        )

    CHROMA_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Rebuild the generated index from the current
    # source documents so deleted/changed files do
    # not leave stale chunks behind.
    if CHROMA_DIR.exists():
        shutil.rmtree(
            CHROMA_DIR
        )

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings = create_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(
            CHROMA_DIR
        ),
        collection_metadata={
            "hnsw:space": "cosine",
        },
    )

    return vector_store


def load_vector_store() -> Chroma:
    """
    Load the existing persistent ChromaDB
    collection using the same embedding model.
    """

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "ChromaDB storage does not exist. "
            "Build the knowledge base first."
        )

    embeddings = create_embedding_model()

    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(
            CHROMA_DIR
        ),
    )