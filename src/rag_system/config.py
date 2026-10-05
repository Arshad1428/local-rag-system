import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

DOCUMENTS_DIR = PROJECT_ROOT / os.getenv(
    "DOCUMENTS_DIR",
    "documents",
)

CHROMA_DIR = PROJECT_ROOT / os.getenv(
    "CHROMA_DIR",
    "storage/chroma",
)

CHROMA_COLLECTION = os.getenv(
    "CHROMA_COLLECTION",
    "local_rag_documents",
)

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "500",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "50",
    )
)

TOP_K = int(
    os.getenv(
        "TOP_K",
        "4",
    )
)


if CHUNK_SIZE <= 0:
    raise ValueError(
        "CHUNK_SIZE must be greater than 0."
    )

if CHUNK_OVERLAP < 0:
    raise ValueError(
        "CHUNK_OVERLAP cannot be negative."
    )

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
    )

if TOP_K <= 0:
    raise ValueError(
        "TOP_K must be greater than 0."
    )