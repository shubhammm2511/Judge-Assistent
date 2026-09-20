from pathlib import Path
from typing import Any

import chromadb

from chunking import chunk_pages
from embeddings import LocalEmbedder
from ingestion import extract_pdf_pages


COLLECTION_NAME = "agent_as_judge_documents"


def get_collection(
    database_path: str | Path,
) -> chromadb.Collection:
    """Create or load the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(database_path))

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_vector_store(
    pdf_path: str | Path,
    database_path: str | Path,
) -> chromadb.Collection:
    """Extract, chunk, embed, and store the PDF content."""
    pages = extract_pdf_pages(pdf_path)
    chunks = chunk_pages(pages)

    embedder = LocalEmbedder()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_documents(texts)

    collection = get_collection(database_path)

    collection.upsert(
        ids=[chunk["metadata"]["chunk_id"] for chunk in chunks],
        documents=texts,
        metadatas=[chunk["metadata"] for chunk in chunks],
        embeddings=embeddings,
    )

    return collection


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    pdf_path = project_root / "data" / "Agent-as-Judge.pdf"
    database_path = project_root / "chroma_db"

    collection = build_vector_store(pdf_path, database_path)

    print(f"Vector store created successfully.")
    print(f"Collection: {collection.name}")
    print(f"Stored chunks: {collection.count()}")