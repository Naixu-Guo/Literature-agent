# Literature Agent - Clean Architecture

## Overview
A streamlined MCP-based literature research agent with document persistence and RAG capabilities.

## Project Structure

```
Literature-agent/
├── main.py                  # Main entry point - starts the API server
├── .env                     # Environment variables (API keys, SECURED)
├── pyproject.toml          # Python project dependencies
│
├── tools/                   # Core implementations
│   ├── literature_agent.py # Main API functions (6 core functions)
│   ├── mcp_backend.py      # MCP backend implementation (internal)
│   └── toolset.py          # Tool registration framework
│
├── mcp_literature_storage/  # Persistent document storage
│   └── *.json              # Stored document metadata and chunks
│
└── storage/                 # Vector database storage
    └── faiss/              # FAISS index files
```

## Core API Functions (6 total)

### 1. Document Input/Processing
- **`summarize_source`** - Load and summarize single or multiple sources
  - Input: `Union[str, List[str]]` (files, URLs, or mixed)
  - Output: Integrated summary for multiple sources

### 2. Document Query
- **`query_documents`** - Query stored documents with two modes
  - `mode="search"` - Find document excerpts
  - `mode="answer"` - AI-generated answers with sources

### 3. Web Research  
- **`web_research`** - Search, download, and summarize web content
  - Auto-downloads PDFs and research papers
  - Stores in MCP for future queries

### 4. Document Management
- **`list_documents`** - List all stored documents
- **`delete_document`** - Remove specific documents  
- **`clear_all_documents`** - Clear all storage

## Security Features
✅ **API Key Protection**: Only in `.env` file (gitignored)
✅ **No Key Exposure**: No hardcoded keys in any code files
✅ **Clean Codebase**: No test/debug functions exposed
✅ **Minimal Attack Surface**: Only 6 essential functions

## API Access
- **Interactive docs**: http://localhost:8001/docs
- **Base URL**: http://localhost:8001

## Environment Variables (Secured in .env)
```bash
GOOGLE_API_KEY=your-secure-api-key
GEMINI_MODEL=gemini-2.5-pro  
EMBED_MODEL=models/embedding-001
LLM_TEMPERATURE=0.2
USER_AGENT=Literature-Agent/1.0
```