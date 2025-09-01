# Literature Review Agent for Quantum Computing

An AI-powered literature review agent built with LangChain and Google Gemini, specialized for quantum computing research.

## Features

### Document Processing
- **PDF Reading**: Load and process PDF research papers
- **Text Splitting**: Intelligent chunking for better context preservation
- **Vector Storage**: FAISS-based vector indexing for efficient retrieval

### Analysis Capabilities
- **Document Summarization**: Generate concise summaries of research papers
- **RAG-based Q&A**: Answer questions based on loaded documents
- **Web Research**: Search for literature via Google Scholar (with safe fallbacks)

### API Endpoints

- `summarize_source(sources, title=None, max_length=800)`
  - Summarize one or multiple sources (local PDFs/text or URLs)
  - For multiple sources, returns an integrated summary using RAG

- `query_documents(query, mode="answer"|"search", num_results=5)`
  - Mode `search`: semantic search excerpts
  - Mode `answer`: RAG answer with sources

- `web_research(query, num_results=5)`
  - Searches Google Scholar and the web, loads and summarizes accessible documents

- `list_documents()` / `delete_document(doc_id)` / `clear_all_documents()`
  - Manage stored documents

## Setup

### Prerequisites
- Python 3.12+
- Google API key with available credits

### Installation

1. Install dependencies:
```bash
pip install -e .
```

2. Set environment variables:
```bash
export GOOGLE_API_KEY="your-google-api-key"
export GEMINI_MODEL="gemini-2.5-pro"  # Optional, defaults to gemini-2.5-pro
export EMBED_MODEL="models/embedding-001"  # Optional
export LLM_TEMPERATURE="0.2"  # Optional, defaults to 0.2
export USER_AGENT="Literature-Agent/1.0"   # Optional
export MAX_DOWNLOAD_BYTES="10485760"       # Optional, 10MB default
export HOST="127.0.0.1"                    # Optional, default binds to localhost
export PORT="8001"                          # Optional
```

### Running the Server

```bash
python main.py
```

The API server will start at `http://localhost:8001`

## Usage Examples

### 1. Summarize from URL or Local File
```python
# Summarize a web page
result = summarize_source("https://arxiv.org/abs/2404.14809")

# Summarize an online PDF
result = summarize_source("https://example.com/paper.pdf")

# Summarize a local file
result = summarize_source("/path/to/paper.pdf")

# Summarize multiple sources at once
results = summarize_multiple_sources([
    "/local/paper.pdf",
    "https://arxiv.org/abs/2404.14809",
    "https://quantum-computing.ibm.com/article"
])
```

### 2. Ask Questions
```python
answer = query_documents("What are the main quantum algorithms discussed?", mode="answer")
```

### 3. Web Research
```python
results = web_research("quantum error correction recent advances", num_results=5)
```

## Important Notes

### API Quotas
- **Google API**: Ensure you have sufficient quota. Check your usage at `https://console.cloud.google.com/`
- If you encounter quota errors, the system will provide helpful error messages.

## Architecture

The agent uses:
- **LangChain**: For document processing and orchestration
- **Google Gemini**: For text generation and understanding
- **FAISS**: For efficient vector similarity search
- **PyPDF**: For PDF document parsing

## API Documentation

The full API documentation is available at `http://localhost:8001/docs` when the server is running.

## License

MIT