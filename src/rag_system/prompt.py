from langchain_core.prompts import (
    ChatPromptTemplate,
)


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a document question-answering assistant.

Answer the user's question using ONLY the provided
document context.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- If the answer is not supported by the context,
  say: "I couldn't find that information in the provided documents."
- Keep the answer focused on the question.
- Do not invent source names or page numbers.
""".strip(),
        ),
        (
            "human",
            """
Document context:

{context}

Question:
{question}

Answer:
""".strip(),
        ),
    ]
)