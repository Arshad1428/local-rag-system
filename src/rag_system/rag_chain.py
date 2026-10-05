from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from .prompt import RAG_PROMPT
from .retriever import retrieve_documents


def create_llm() -> ChatOllama:
    """
    Create the local Ollama chat model.
    """

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )


def format_context(
    documents: list[Document],
) -> str:
    """
    Format retrieved chunks for prompt injection.
    """

    if not documents:
        return "No relevant document context was retrieved."

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        source = document.metadata.get(
            "source",
            "unknown",
        )

        page = document.metadata.get(
            "page",
        )

        if page is not None:
            location = (
                f"{source}, page {page + 1}"
            )
        else:
            location = source

        context_parts.append(
            f"[Source {index}: {location}]\n"
            f"{document.page_content.strip()}"
        )

    return "\n\n".join(
        context_parts
    )


def build_sources(
    documents: list[Document],
) -> list[dict]:
    """
    Build deduplicated source references from the
    retrieved LangChain Documents.
    """

    sources = []
    seen = set()

    for document in documents:
        source = document.metadata.get(
            "source",
            "unknown",
        )

        page = document.metadata.get(
            "page",
        )

        key = (
            source,
            page,
        )

        if key in seen:
            continue

        seen.add(key)

        source_record = {
            "source": source,
        }

        if page is not None:
            source_record["page"] = (
                page + 1
            )

        sources.append(
            source_record
        )

    return sources


def ask_question(
    vector_store: Chroma,
    question: str,
) -> dict:
    """
    Run the complete RAG workflow:

    question
        -> retrieval
        -> context injection
        -> Ollama generation
        -> answer + sources
    """

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    retrieved_documents = (
        retrieve_documents(
            vector_store=vector_store,
            query=question,
        )
    )

    context = format_context(
        retrieved_documents
    )

    prompt_value = RAG_PROMPT.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    llm = create_llm()

    response = llm.invoke(
        prompt_value
    )

    return {
        "answer": response.content.strip(),
        "sources": build_sources(
            retrieved_documents
        ),
        "retrieved_documents": (
            retrieved_documents
        ),
    }