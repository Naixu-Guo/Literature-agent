# Literature Agent Architecture

## Overview
A unified MCP-based literature research agent with document persistence and RAG capabilities.

## Project Structure

```
Literature-agent/
├── main.py                  # Main entry point - starts the API server
├── .env                     # Environment variables (API keys, model settings)
├── pyproject.toml          # Python project dependencies
│
├── tools/                   # Tool implementations
│   ├── __init__.py
│   ├── toolset.py          # Tool registration framework
│   ├── literature_agent.py # Main API functions (unified MCP-based)
│   ├── mcp_backend.py      # MCP backend implementation (internal)
│   └── calculator.py       # Simple calculator tools for testing
│
├── mcp_literature_storage/  # Persistent document storage
│   └── *.json              # Stored document metadata and chunks
│
└── storage/                 # Vector database storage
    └── faiss/              # FAISS index files
        ├── index.faiss
        └── index.pkl

```

## Architecture

### 1. API Layer (`literature_agent.py`)
All user-facing functions using unified MCP backend:
- `summarize_source` - Load and summarize documents
- `summarize_multiple_sources` - Batch summarization with combined insights
- `build_vector_index` - Build searchable index from documents
- `rag_answer` - Answer questions using RAG
- `web_research` - Search and load web documents
- `search_documents` - Search across stored documents
- `list_documents` - List all stored documents
- `delete_document` - Remove specific documents
- `clear_all_documents` - Clear all storage

### 2. MCP Backend (`mcp_backend.py`)
Internal document management system:
- Document loading (PDF, text, web)
- Persistent storage with JSON metadata
- Vector embeddings with FAISS
- LLM integration (Google Gemini)
- Document chunking and retrieval

### 3. Benefits of Unified MCP Architecture
- **Performance**: Documents loaded once, reused across operations
- **Persistence**: Documents survive server restarts
- **Efficiency**: Shared vector index for all RAG operations
- **Consistency**: Single source of truth for document storage
- **Scalability**: Can handle large document collections

## API Access
- Interactive docs: http://localhost:8001/docs
- Base URL: http://localhost:8001

## Environment Variables
- `GOOGLE_API_KEY`: Required for Gemini LLM
- `GEMINI_MODEL`: Model name (default: gemini-2.5-pro)
- `EMBED_MODEL`: Embedding model (default: models/embedding-001)
- `LLM_TEMPERATURE`: Model temperature (default: 0.2)
- `USER_AGENT`: User agent for web requests