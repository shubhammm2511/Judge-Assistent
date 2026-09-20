import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from retriever import retrieve_chunks


load_dotenv()


RAG_SYSTEM_PROMPT = """You are a careful research-paper question-answering assistant.

Start with the direct answer. Do not begin with phrases such as
"Based on the retrieved context" or "According to the retrieved context."
The retrieved sources are displayed separately by the application.

Pay careful attention to units and labels.
Never swap time percentages with cost percentages.
For time, use hours or minutes.
For cost, use dollar amounts.
When the context provides raw values and percentages, preserve the correct relationship between them.
If a percentage is derived from the retrieved values, calculate and label it explicitly.

Use the retrieved context as the primary and authoritative source.
Answer only from the provided context.
Do not invent facts, numbers, names, or explanations.
If the answer is not supported by the context, say:
"The retrieved context does not contain enough information to answer this question."

Preserve important numbers and technical terminology.
If the question asks for multiple values, explicitly provide every requested value.
Give a concise, complete, factual answer.

When multiple retrieved passages appear inconsistent, prefer the passage
that directly and explicitly answers the user's question over an inferred
calculation from raw values. If the retrieved context contains both a summary
claim and a detailed analysis, use the explicit summary claim for the direct
answer and mention the detailed figures only as supporting context.
If a contradiction remains, state it clearly with the relevant page numbers.

Do not mention information from outside the retrieved context.
"""


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read local environment variables or Streamlit Cloud secrets."""
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:
        return default


def get_nvidia_client() -> OpenAI:
    api_key = get_setting("NVIDIA_API_KEY")
    base_url = get_setting(
        "NVIDIA_BASE_URL",
        "https://integrate.api.nvidia.com/v1",
    )

    if not api_key or api_key == "PASTE_YOUR_NVIDIA_KEY_HERE":
        raise ValueError("NVIDIA_API_KEY is missing or still contains the placeholder.")

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def build_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk["metadata"]

        context_parts.append(
            f"""Chunk {index}
Source: {metadata["source"]}
Page: {metadata["page"]}
Similarity: {chunk["similarity"]:.4f}

{chunk["text"]}
"""
        )

    return "\n\n".join(context_parts)


def build_messages(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    context = build_context(retrieved_chunks)

    user_prompt = f"""Retrieved context:

{context}

Question:
{question}

Answer using only the retrieved context.
"""

    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def stream_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
):
    """Stream answer text from NVIDIA for an already retrieved context."""
    client = get_nvidia_client()
    model = get_setting(
        "NVIDIA_MODEL",
        "nvidia/nemotron-3-ultra-550b-a55b",
    )

    response = client.chat.completions.create(
        model=model,
        messages=build_messages(question, retrieved_chunks),
        temperature=0.0,
        max_tokens=1000,
        stream=True,
    )

    for event in response:
        if not event.choices:
            continue

        content = event.choices[0].delta.content

        if content:
            yield content


def answer_question(
    question: str,
    database_path: str | Path,
    top_k: int = 4,
) -> dict[str, Any]:
    retrieved_chunks = retrieve_chunks(
        question=question,
        database_path=database_path,
        top_k=top_k,
    )

    client = get_nvidia_client()

    model = get_setting(
        "NVIDIA_MODEL",
        "nvidia/nemotron-3-ultra-550b-a55b",
    )

    response = client.chat.completions.create(
        model=model,
        messages=build_messages(question, retrieved_chunks),
        temperature=0.0,
        max_tokens=1000,
    )

    answer = response.choices[0].message.content

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "model": model,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    database_path = project_root / "chroma_db"

    question = "What is the DevAI dataset?"

    result = answer_question(
        question=question,
        database_path=database_path,
        top_k=4,
    )

    print("QUESTION:")
    print(result["question"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nRETRIEVED SOURCES:")

    for index, chunk in enumerate(result["retrieved_chunks"], start=1):
        metadata = chunk["metadata"]

        print(
            f"{index}. {metadata['source']}, "
            f"page {metadata['page']}, "
            f"similarity {chunk['similarity']:.4f}"
        )
