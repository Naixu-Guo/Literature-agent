# Literature Review Agent for Quantum Computing

An AI-powered literature review agent built with LangChain and OpenAI, specialized for quantum computing research.

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

#### 1. `summarize_file(path, temperature)`
Summarize a local PDF or text file with key findings, methods, and limitations.

#### 2. `build_vector_index(paths)`
Build a FAISS vector index from multiple documents for efficient retrieval.

#### 3. `rag_answer(question, index_dir, k, temperature)`
Answer questions using retrieval-augmented generation over indexed documents.

#### 4. `web_research(query, num_results)`
Search the web for quantum computing research and return formatted results.

## Setup

### Prerequisites
- Python 3.12+
- OpenAI API key

### Installation

1. Install dependencies:
```bash
pip install -e .
```

2. Set environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-4o-mini"  # Optional, defaults to gpt-4o-mini
export EMBED_MODEL="text-embedding-3-small"  # Optional
export LLM_TEMPERATURE="0.2"  # Optional, defaults to 0.2
```

### Running the Server

```bash
python main.py
```

The API server will start at `http://localhost:8000`

## Usage Examples

### 1. Summarize a PDF
```python
result = summarize_file("/path/to/paper.pdf")
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

## Architecture

The agent uses:
- **LangChain**: For document processing and chain orchestration
- **OpenAI GPT**: For text generation and understanding
- **FAISS**: For efficient vector similarity search
- **PyPDF**: For PDF document parsing
- **DuckDuckGo Search**: For web research capabilities

## API Documentation

The full API documentation is available at `http://localhost:8000/docs` when the server is running.

## License

MIT