from pathlib import Path
from statistics import mean
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion import extract_pdf_pages


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> list[dict[str, Any]]:
    """
    Split page text into overlapping chunks while preserving page metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        source = page["metadata"]["source"]
        page_number = page["metadata"]["page"]

        for chunk_index, chunk_text in enumerate(page_chunks):
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "source": source,
                        "page": page_number,
                        "chunk_id": (
                            f"{Path(source).stem}"
                            f"-page-{page_number}"
                            f"-chunk-{chunk_index + 1}"
                        ),
                    },
                }
            )

    return chunks


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    pdf_path = project_root / "data" / "Agent-as-Judge.pdf"

    pages = extract_pdf_pages(pdf_path)
    chunks = chunk_pages(pages)

    chunk_lengths = [len(chunk["text"]) for chunk in chunks]

    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Minimum chunk length: {min(chunk_lengths)}")
    print(f"Maximum chunk length: {max(chunk_lengths)}")
    print(f"Average chunk length: {mean(chunk_lengths):.2f}")

    print("\nFirst chunk metadata:")
    print(chunks[0]["metadata"])

    print("\nFirst chunk preview:")
    print(chunks[0]["text"][:500])