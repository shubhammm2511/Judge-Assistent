You can open the deployed chatbot here: [Judge Assistant](https://judge-assistent-uw.streamlit.app/)

# Agent-as-a-Judge RAG Assistant

A retrieval-augmented generation chatbot for the research paper **Agent-as-a-Judge: Evaluate Agents with Agents** by Mingchen Zhuge et al.

The application retrieves relevant passages from the PDF before asking a language model to answer. It displays the answer together with retrieved chunks, source pages, and similarity scores.

## Architecture

```text
PDF
  → Text extraction with PyMuPDF
  → Cleaning and chunking
  → Page metadata
  → Sentence-transformer embeddings
  → ChromaDB vector store
  → Question embedding
  → Similarity retrieval
  → Grounded prompt
  → NVIDIA OpenAI-compatible API
  → Answer with retrieved sources
```

## Technologies

- Python 3.11+
- PyMuPDF
- LangChain text splitters
- Sentence Transformers
- ChromaDB
- NVIDIA OpenAI-compatible API
- Streamlit

## Project structure

```text
data/
└── Agent-as-Judge.pdf
src/
├── ingestion.py
├── chunking.py
├── embeddings.py
├── vector_store.py
├── retriever.py
├── rag_pipeline.py
└── app.py
visualization/
└── rag_workflow.png
tests/
└── test_pipeline.py
requirements.txt
README.md
.gitignore
```

Local-only files such as `.env`, `.venv`, and `chroma_db/` are not committed.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a local `.env` file:

```env
NVIDIA_API_KEY=your_nvidia_api_key
NVIDIA_MODEL=nvidia/nemotron-3-ultra-550b-a55b
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```

## Run the chatbot

```powershell
streamlit run .\src\app.py
```

The application builds the vector index automatically on first startup and displays retrieved evidence with page numbers and similarity scores.

## Deploy on Streamlit Community Cloud

Use:

```text
Repository: shubhammm2511/Judge-Assistent
Branch: main
Main file: src/app.py
```

Add these values to Streamlit Cloud Secrets:

```toml
NVIDIA_API_KEY = "your_nvidia_api_key"
NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
```

## Testing

```powershell
python -m unittest discover -s tests -v
```

## Security

Never commit `.env` or expose the NVIDIA API key. Use Streamlit Cloud Secrets for deployment.
