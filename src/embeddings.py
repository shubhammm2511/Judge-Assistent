from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from chunking import chunk_pages
from ingestion import extract_pdf_pages


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbedder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        return embedding.tolist()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    pdf_path = project_root / "data" / "Agent-as-Judge.pdf"

    pages = extract_pdf_pages(pdf_path)
    chunks = chunk_pages(pages)

    embedder = LocalEmbedder()

    sample_texts = [chunk["text"] for chunk in chunks[:3]]
    sample_embeddings = embedder.embed_documents(sample_texts)

    print(f"Embedded texts: {len(sample_embeddings)}")
    print(f"Embedding dimension: {len(sample_embeddings[0])}")
    print(f"First embedding values: {sample_embeddings[0][:5]}")