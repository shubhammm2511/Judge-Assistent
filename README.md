You can open the deployed chatbot here: (https://judge-assistent-uw.streamlit.app/)

# Agent-as-a-Judge RAG Assistant

A retrieval-augmented generation chatbot for the research paper **Agent-as-a-Judge: Evaluate Agents with Agents** by Mingchen Zhuge et al.

The application retrieves relevant passages from the PDF, sends only those passages to the language model, and displays the answer together with source pages, retrieved chunks, and similarity scores.

## Architecture

```text
PDF
  → Text extraction with PyMuPDF
  → Cleaning and chunking
  → Page metadata
  → Sentence-transformer embeddings
  → ChromaDB vector store
  → Question embedding
  → Similarity and lexical retrieval
  → Grounded prompt
  → OpenRouter LLM
  → Answer with retrieved sources
```

## Technologies

- Python 3.11+
- PyMuPDF
- LangChain text splitters
- Sentence Transformers
- ChromaDB
- Streamlit
- OpenRouter API

## Project structure

```text
data/Agent-as-Judge.pdf
src/
├── ingestion.py
├── chunking.py
├── embeddings.py
├── vector_store.py
├── retriever.py
├── rag_pipeline.py
└── app.py
visualization/rag_workflow.png
tests/test_pipeline.py
requirements.txt
README.md
.gitignore
```


## Limitations

- The system currently indexes one PDF.
- PDF extraction may be imperfect for tables and figures.
- Retrieval quality depends on chunking and embedding quality.
- The free model may have rate limits or variable response times.
- Similarity scores indicate retrieval relevance; they are not truth probabilities.
