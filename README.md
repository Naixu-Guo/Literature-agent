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
- **Web Research**: Search for quantum computing literature online via DuckDuckGo

### API Endpoints

#### 1. `summarize_source(source, temperature)`
Summarize content from local files OR URLs (PDFs, text files, web pages).
- Supports: Local PDFs, text files, web URLs, online PDF URLs
- Returns: Comprehensive summary with key findings


#### 2. `summarize_multiple_sources(sources, temperature)`
Summarize multiple sources at once and get both individual and combined summaries.
- Input: List of file paths and/or URLs
- Returns: Individual summaries + synthesized overview

#### 3. `build_vector_index(paths)`
Build a FAISS vector index from multiple documents for efficient retrieval.

#### 4. `rag_answer(question, index_dir, k, temperature)`
Answer questions using retrieval-augmented generation over indexed documents.

#### 5. `web_research(query, num_results)`
Search for quantum computing research using Google Gemini's knowledge base with academic focus.

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

### 2. Build Vector Index
```python
index_path = build_vector_index([
    "/path/to/paper1.pdf",
    "/path/to/paper2.pdf"
])
```

### 3. Ask Questions
```python
answer = rag_answer("What are the main quantum algorithms discussed?")
```

### 4. Web Research
```python
results = web_research("quantum error correction recent advances", num_results=5)
```

## Important Notes

### API Quotas
- **Google API**: Ensure you have sufficient quota. Check your usage at https://console.cloud.google.com/
- If you encounter quota errors, the system will provide helpful error messages.

## Architecture

The agent uses:
- **LangChain**: For document processing and chain orchestration
- **Google Gemini**: For text generation and understanding
- **FAISS**: For efficient vector similarity search
- **PyPDF**: For PDF document parsing
- **Gemini Knowledge Base**: For web research capabilities

## API Documentation

The full API documentation is available at `http://localhost:8001/docs` when the server is running.

## License

MIT