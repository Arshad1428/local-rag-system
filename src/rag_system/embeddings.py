from langchain_huggingface import (
    HuggingFaceEmbeddings,
)

from .config import EMBEDDING_MODEL


def create_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create the local embedding model used for
    both indexing documents and embedding queries.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )