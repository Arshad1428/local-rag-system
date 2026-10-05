import streamlit as st

from src.rag_system.chunker import split_documents
from src.rag_system.document_loader import load_documents
from src.rag_system.rag_chain import ask_question
from src.rag_system.vector_store import (
    create_vector_store,
    load_vector_store,
)


st.set_page_config(
    page_title="Local RAG System",
    page_icon="📚",
    layout="wide",
)


st.title(
    "📚 Local RAG Document Q&A"
)

st.write(
    """
    Ask questions about your local documents using
    **LangChain, ChromaDB, and Ollama**.
    """
)


# --------------------------------------------------
# Sidebar — knowledge-base management
# --------------------------------------------------

with st.sidebar:
    st.header(
        "Knowledge Base"
    )

    st.write(
        """
        Add PDF or TXT files to the `documents`
        directory, then build the knowledge base.
        """
    )

    if st.button(
        "Build Knowledge Base",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Loading and indexing documents..."
            ):
                documents = load_documents()

                chunks = split_documents(
                    documents
                )

                vector_store = (
                    create_vector_store(
                        chunks
                    )
                )

                chunk_count = (
                    vector_store
                    ._collection
                    .count()
                )

            st.success(
                "Knowledge base created."
            )

            st.metric(
                "Loaded records",
                len(documents),
            )

            st.metric(
                "Stored chunks",
                chunk_count,
            )

        except Exception as error:
            st.error(
                "Failed to build the knowledge base."
            )

            st.exception(error)


# --------------------------------------------------
# Question answering
# --------------------------------------------------

st.subheader(
    "Ask Your Documents"
)

question = st.text_input(
    "Question",
    placeholder=(
        "Example: What does the document say "
        "about vector databases?"
    ),
)


if st.button(
    "Ask",
    type="primary",
):
    if not question.strip():
        st.warning(
            "Enter a question."
        )

    else:
        try:
            with st.spinner(
                "Retrieving context and generating answer..."
            ):
                vector_store = (
                    load_vector_store()
                )

                result = ask_question(
                    vector_store=vector_store,
                    question=question,
                )

            # ------------------------------------------
            # Generated answer
            # ------------------------------------------

            st.markdown(
                "## Answer"
            )

            st.write(
                result["answer"]
            )

            # ------------------------------------------
            # Sources
            # ------------------------------------------

            st.markdown(
                "## Sources"
            )

            if result["sources"]:
                for source in result["sources"]:
                    source_name = (
                        source["source"]
                    )

                    page = source.get(
                        "page"
                    )

                    if page is not None:
                        st.write(
                            f"- {source_name} — page {page}"
                        )
                    else:
                        st.write(
                            f"- {source_name}"
                        )

            else:
                st.info(
                    "No sources were retrieved."
                )

            # ------------------------------------------
            # Retrieved chunks
            # ------------------------------------------

            with st.expander(
                "Retrieved Context"
            ):
                for index, document in enumerate(
                    result[
                        "retrieved_documents"
                    ],
                    start=1,
                ):
                    st.markdown(
                        f"### Chunk {index}"
                    )

                    st.write(
                        document.page_content
                    )

                    source = (
                        document.metadata.get(
                            "source",
                            "unknown",
                        )
                    )

                    page = (
                        document.metadata.get(
                            "page"
                        )
                    )

                    if page is not None:
                        st.caption(
                            f"{source} — page {page + 1}"
                        )
                    else:
                        st.caption(
                            source
                        )

                    st.divider()

        except FileNotFoundError:
            st.error(
                "The knowledge base has not been "
                "built yet. Add documents and use "
                "'Build Knowledge Base' first."
            )

        except Exception as error:
            st.error(
                "RAG request failed."
            )

            st.exception(error)