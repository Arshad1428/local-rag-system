from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from .config import DOCUMENTS_DIR


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
}


def load_document(
    file_path: Path,
) -> list[Document]:
    """
    Load one supported document.

    PDF files are loaded page by page.
    TXT files are loaded as text documents.
    """

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(
            str(file_path)
        )

    elif suffix == ".txt":
        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
            autodetect_encoding=True,
        )

    else:
        raise ValueError(
            f"Unsupported document type: {suffix}"
        )

    documents = loader.load()

    for document in documents:
        document.metadata["source"] = (
            file_path.name
        )

        document.metadata["file_type"] = (
            suffix.lstrip(".")
        )

    return [
        document
        for document in documents
        if document.page_content.strip()
    ]


def load_documents(
    documents_dir: Path = DOCUMENTS_DIR,
) -> list[Document]:
    """
    Load every supported document from the
    configured documents directory.
    """

    if not documents_dir.exists():
        raise FileNotFoundError(
            "Documents directory not found: "
            f"{documents_dir}"
        )

    files = sorted(
        file_path
        for file_path in documents_dir.rglob("*")
        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    )

    if not files:
        raise ValueError(
            "No PDF or TXT documents were found in "
            f"{documents_dir}"
        )

    documents: list[Document] = []

    for file_path in files:
        documents.extend(
            load_document(
                file_path
            )
        )

    if not documents:
        raise ValueError(
            "The document files contained no "
            "searchable text."
        )

    return documents