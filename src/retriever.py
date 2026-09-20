import re
from pathlib import Path
from typing import Any

from embeddings import LocalEmbedder
from vector_store import get_collection


STOPWORDS = {
    "a", "an", "and", "are", "as", "by", "compared", "does", "for",
    "how", "in", "is", "of", "on", "the", "to", "was", "what", "with",
}


def tokenize(text: str) -> set[str]:
    """Extract meaningful lowercase word and number tokens."""
    tokens = re.findall(r"[a-zA-Z0-9.%]+", text.lower())
    return {token for token in tokens if token not in STOPWORDS}


def normalized_bigrams(text: str) -> set[tuple[str, str]]:
    tokens = re.findall(r"[a-zA-Z0-9.%]+", text.lower())
    tokens = [token for token in tokens if token not in STOPWORDS]
    return set(zip(tokens, tokens[1:]))


def retrieve_chunks(
    question: str,
    database_path: str | Path,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Retrieve chunks using semantic similarity plus keyword overlap."""
    embedder = LocalEmbedder()
    query_embedding = embedder.embed_query(question)

    collection = get_collection(database_path)
    total_chunks = collection.count()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=total_chunks,
        include=["documents", "metadatas", "distances"],
    )

    query_tokens = tokenize(question)
    query_bigrams = normalized_bigrams(question)
    retrieved_chunks = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        semantic_similarity = 1.0 - float(distance)
        document_tokens = tokenize(document)
        lexical_score = len(query_tokens & document_tokens) / max(
            len(query_tokens), 1
        )
        document_bigrams = normalized_bigrams(document)
        phrase_score = len(query_bigrams & document_bigrams) / max(
            len(query_bigrams), 1
        )
        hybrid_score = (
            0.4 * semantic_similarity
            + 0.35 * lexical_score
            + 0.25 * phrase_score
        )

        retrieved_chunks.append(
            {
                "text": document,
                "metadata": metadata,
                "distance": float(distance),
                "similarity": semantic_similarity,
                "lexical_score": lexical_score,
                "phrase_score": phrase_score,
                "hybrid_score": hybrid_score,
            }
        )

    retrieved_chunks.sort(
        key=lambda result: result["hybrid_score"],
        reverse=True,
    )

    return retrieved_chunks[:top_k]


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    database_path = project_root / "chroma_db"

    test_question = (
        "What percentage of evaluation time and cost does "
        "Agent-as-a-Judge save compared to using three human experts?"
    )
    results = retrieve_chunks(
        question=test_question,
        database_path=database_path,
        top_k=8,
    )

    print(f"Question: {test_question}")
    print(f"Retrieved chunks: {len(results)}")

    for index, result in enumerate(results, start=1):
        print("\n" + "=" * 70)
        print(f"Result {index}")
        print(f"Metadata: {result['metadata']}")
        print(f"Semantic similarity: {result['similarity']:.4f}")
        print(f"Lexical score: {result['lexical_score']:.4f}")
        print(f"Phrase score: {result['phrase_score']:.4f}")
        print(f"Hybrid score: {result['hybrid_score']:.4f}")
        print(f"Text:\n{result['text'][:700]}")
