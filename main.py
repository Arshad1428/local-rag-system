import sys

from src.rag_system.chunker import split_documents
from src.rag_system.document_loader import load_documents
from src.rag_system.rag_chain import ask_question
from src.rag_system.vector_store import (
    create_vector_store,
    load_vector_store,
)


def ingest_documents() -> None:
    """Build or rebuild the local RAG knowledge base."""

    print("\nBuilding RAG knowledge base...")

    documents = load_documents()

    print(
        f"Loaded {len(documents)} document/page records."
    )

    chunks = split_documents(documents)

    print(
        f"Created {len(chunks)} chunks."
    )

    vector_store = create_vector_store(chunks)

    chunk_count = (
        vector_store
        ._collection
        .count()
    )

    print(
        f"Stored {chunk_count} chunks in ChromaDB."
    )

    print(
        "Knowledge base created successfully."
    )


def ask(question: str) -> None:
    """Ask a question against the knowledge base."""

    vector_store = load_vector_store()

    result = ask_question(
        vector_store=vector_store,
        question=question,
    )

    print("\nAnswer")
    print("-" * 60)
    print(result["answer"])

    print("\nSources")
    print("-" * 60)

    if not result["sources"]:
        print("No sources retrieved.")
        return

    for source in result["sources"]:
        source_name = source["source"]
        page = source.get("page")

        if page is not None:
            print(
                f"- {source_name}, page {page}"
            )
        else:
            print(
                f"- {source_name}"
            )


def print_usage() -> None:
    print(
        """
Usage:

Build or rebuild the knowledge base:
    python main.py ingest

Ask a question:
    python main.py ask "your question"
""".strip()
    )


def main() -> None:
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].strip().lower()

    if command == "ingest":
        ingest_documents()
        return

    if command == "ask":
        if len(sys.argv) < 3:
            print(
                "Please provide a question."
            )
            return

        question = " ".join(
            sys.argv[2:]
        )

        ask(question)
        return

    print(
        f"Unknown command: {command}"
    )

    print_usage()


if __name__ == "__main__":
    main()