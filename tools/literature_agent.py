"""
Unified MCP-based Literature Agent
All functions use the MCP backend for optimal performance and consistency
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from tools.toolset import toolset
from tools.mcp_backend import _agent


@toolset.add()
def summarize_source(source: str, title: Optional[str] = None, max_length: int = 800) -> str:
    """
    Load and summarize content from a local file or URL using MCP backend.
    Documents are automatically stored for future operations.
    
    :param source: absolute path to a local file (.pdf or .txt) or URL to a web page/PDF
    :param title: optional title for the document
    :param max_length: maximum length of the summary (default: 800)
    :return: summary string
    """
    try:
        # Convert arXiv abstract URLs to PDF URLs for better loading
        if 'arxiv.org/abs/' in source:
            source = source.replace('/abs/', '/pdf/') + '.pdf'
        elif 'journals.aps.org/' in source and '/abstract/' in source:
            source = source.replace('/abstract/', '/pdf/')
        
        # Load document using MCP backend
        if source.startswith(('http://', 'https://')):
            load_result = _agent.load_url(source, title=title)
        else:
            if not os.path.isfile(source):
                return f"File not found: {source}"
            load_result = _agent.load_file(source, title=title)
        
        if "error" in load_result:
            return f"Error loading document: {load_result['error']}"
        
        if not load_result.get("success"):
            return "Failed to load document"
        
        # Get document ID
        doc_id = load_result["doc_id"]
        
        # Generate summary using MCP backend
        summary_result = _agent.summarize(doc_id, max_length=max_length)
        
        if "error" in summary_result:
            return f"Error generating summary: {summary_result['error']}"
        
        return summary_result.get("summary", "No summary available")
        
    except Exception as e:
        return f"Error: {str(e)}"


@toolset.add()
def summarize_multiple_sources(sources: list, max_length_per_source: int = 500) -> dict:
    """
    Load and summarize multiple sources using MCP backend.
    All documents are stored for future operations.
    
    :param sources: list of file paths or URLs
    :param max_length_per_source: maximum length per summary (default: 500)
    :return: dictionary with combined summary and individual summaries
    """
    try:
        summaries = {}
        doc_ids = []
        
        # Load and summarize each source
        for source in sources:
            # Convert URLs if needed
            if 'arxiv.org/abs/' in source:
                source = source.replace('/abs/', '/pdf/') + '.pdf'
            elif 'journals.aps.org/' in source and '/abstract/' in source:
                source = source.replace('/abstract/', '/pdf/')
            
            # Load document
            if source.startswith(('http://', 'https://')):
                load_result = _agent.load_url(source)
            else:
                if not os.path.isfile(source):
                    summaries[source] = f"File not found: {source}"
                    continue
                load_result = _agent.load_file(source)
            
            if "error" in load_result:
                summaries[source] = f"Error loading: {load_result['error']}"
                continue
            
            if not load_result.get("success"):
                summaries[source] = "Failed to load document"
                continue
            
            doc_id = load_result["doc_id"]
            doc_ids.append(doc_id)
            
            # Summarize
            summary_result = _agent.summarize(doc_id, max_length=max_length_per_source)
            if "error" in summary_result:
                summaries[source] = f"Error: {summary_result['error']}"
            else:
                summaries[source] = summary_result.get("summary", "No summary available")
        
        # Generate combined summary using RAG across all documents
        if doc_ids:
            combined_question = "Provide a comprehensive summary of the main themes, findings, and insights across all these documents."
            combined_result = _agent.ask_question(combined_question, num_sources=len(doc_ids))
            
            if "error" in combined_result:
                combined_summary = f"Error generating combined summary: {combined_result['error']}"
            else:
                combined_summary = combined_result.get("answer", "No combined summary available")
        else:
            combined_summary = "No documents were successfully loaded."
        
        return {
            "combined_summary": combined_summary,
            "individual_summaries": summaries
        }
        
    except Exception as e:
        return {
            "combined_summary": f"Error: {str(e)}",
            "individual_summaries": {}
        }


@toolset.add()
def build_vector_index(sources: list) -> str:
    """
    Build a vector index from multiple documents using MCP backend.
    Documents are loaded and stored with embeddings for RAG operations.
    
    :param sources: list of file paths or URLs to index
    :return: status message
    """
    try:
        successful = 0
        failed = 0
        
        for source in sources:
            # Convert URLs if needed
            if 'arxiv.org/abs/' in source:
                source = source.replace('/abs/', '/pdf/') + '.pdf'
            elif 'journals.aps.org/' in source and '/abstract/' in source:
                source = source.replace('/abstract/', '/pdf/')
            
            # Load document
            if source.startswith(('http://', 'https://')):
                load_result = _agent.load_url(source)
            else:
                if not os.path.isfile(source):
                    failed += 1
                    continue
                load_result = _agent.load_file(source)
            
            if load_result.get("success"):
                successful += 1
            else:
                failed += 1
        
        # The MCP backend automatically builds embeddings when documents are loaded
        status = f"Vector index built: {successful} documents indexed successfully"
        if failed > 0:
            status += f", {failed} documents failed"
        
        # List all documents to confirm
        docs = _agent.list_documents()
        if docs.get("documents"):
            status += f". Total documents in index: {len(docs['documents'])}"
        
        return status
        
    except Exception as e:
        return f"Error building vector index: {str(e)}"


@toolset.add()
def rag_answer(question: str, num_sources: int = 5) -> str:
    """
    Answer a question using RAG over all indexed documents in MCP storage.
    
    :param question: the question to answer
    :param num_sources: number of source chunks to retrieve (default: 5)
    :return: the answer with sources
    """
    try:
        # Use MCP backend's ask_question which performs RAG
        result = _agent.ask_question(question, num_sources=num_sources)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        answer = result.get("answer", "No answer available")
        sources = result.get("sources", [])
        
        # Format response with sources
        response = f"{answer}\n"
        if sources:
            response += "\n\nSources:\n"
            for i, source in enumerate(sources, 1):
                response += f"{i}. {source}\n"
        
        return response
        
    except Exception as e:
        return f"Error: {str(e)}"


@toolset.add()
def web_research(query: str, num_results: int = 5) -> list:
    """
    Search and load web documents related to a query using MCP backend.
    
    :param query: search query
    :param num_results: maximum number of results to load (default: 5)
    :return: list of summaries from loaded documents
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import quote
        
        # Get user agent from environment
        user_agent = os.getenv("USER_AGENT", "Literature-Agent/1.0")
        
        # Search using Google Scholar
        search_url = f"https://scholar.google.com/scholar?q={quote(query)}&hl=en"
        headers = {'User-Agent': user_agent}
        
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        loaded_count = 0
        
        # Find result divs
        for result_div in soup.find_all('div', {'class': 'gs_ri'}):
            if loaded_count >= num_results:
                break
            
            # Extract title and URL
            title_elem = result_div.find('h3', {'class': 'gs_rt'})
            if not title_elem:
                continue
            
            link_elem = title_elem.find('a')
            if not link_elem or not link_elem.get('href'):
                continue
            
            title = title_elem.get_text(strip=True)
            url = link_elem['href']
            
            # Skip citations and non-accessible content
            if '[CITATION]' in title or '[HTML]' in title:
                continue
            
            # Extract snippet
            snippet_elem = result_div.find('div', {'class': 'gs_rs'})
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
            
            # Try to load the document using MCP
            load_result = _agent.load_url(url, title=title)
            
            if load_result.get("success"):
                doc_id = load_result["doc_id"]
                
                # Generate summary
                summary_result = _agent.summarize(doc_id, max_length=300)
                
                if not summary_result.get("error"):
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "summary": summary_result.get("summary", "No summary available"),
                        "doc_id": doc_id
                    })
                    loaded_count += 1
        
        # If no Scholar results, try regular web search
        if not results:
            search_url = f"https://www.google.com/search?q={quote(query + ' filetype:pdf OR research paper')}"
            response = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                if loaded_count >= num_results:
                    break
                
                href = link['href']
                if '/url?q=' in href and '.pdf' in href:
                    url = href.split('/url?q=')[1].split('&')[0]
                    
                    # Try to load PDF
                    load_result = _agent.load_url(url)
                    
                    if load_result.get("success"):
                        doc_id = load_result["doc_id"]
                        summary_result = _agent.summarize(doc_id, max_length=300)
                        
                        if not summary_result.get("error"):
                            results.append({
                                "title": f"Document from {url.split('/')[2]}",
                                "url": url,
                                "summary": summary_result.get("summary", "No summary available"),
                                "doc_id": doc_id
                            })
                            loaded_count += 1
        
        return results
        
    except Exception as e:
        return [{"error": str(e)}]


@toolset.add()
def search_documents(query: str, num_results: int = 5) -> str:
    """
    Search across all stored documents in MCP storage.
    
    :param query: search query
    :param num_results: number of results to return (default: 5)
    :return: search results with relevant excerpts
    """
    try:
        result = _agent.search(query, num_results=num_results)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        matches = result.get("results", [])
        
        if not matches:
            return "No matching documents found."
        
        # Format results
        output = f"Found {len(matches)} relevant documents:\n\n"
        for i, match in enumerate(matches, 1):
            output += f"{i}. {match}\n\n"
        
        return output
        
    except Exception as e:
        return f"Error: {str(e)}"


@toolset.add()
def list_documents() -> str:
    """
    List all documents currently stored in MCP storage.
    
    :return: formatted list of documents
    """
    try:
        result = _agent.list_documents()
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        docs = result.get("documents", [])
        
        if not docs:
            return "No documents in storage."
        
        # Format document list
        output = f"Documents in storage ({len(docs)} total):\n\n"
        for doc in docs:
            output += f"ID: {doc['id']}\n"
            output += f"Title: {doc['title']}\n"
            output += f"Source: {doc['source']}\n"
            output += f"Loaded: {doc['loaded_at']}\n"
            output += f"Chunks: {doc['num_chunks']}\n"
            output += "-" * 50 + "\n"
        
        return output
        
    except Exception as e:
        return f"Error: {str(e)}"


@toolset.add()
def delete_document(doc_id: str) -> str:
    """
    Delete a document from MCP storage.
    
    :param doc_id: the document ID to delete
    :return: status message
    """
    try:
        result = _agent.delete_document(doc_id)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if result.get("success"):
            return f"Document {doc_id} deleted successfully."
        else:
            return f"Failed to delete document {doc_id}."
        
    except Exception as e:
        return f"Error: {str(e)}"


@toolset.add()
def clear_all_documents() -> str:
    """
    Clear all documents from MCP storage.
    
    :return: status message
    """
    try:
        # Get all documents
        list_result = _agent.list_documents()
        docs = list_result.get("documents", [])
        
        if not docs:
            return "No documents to clear."
        
        # Delete each document
        deleted = 0
        failed = 0
        
        for doc in docs:
            result = _agent.delete_document(doc["id"])
            if result.get("success"):
                deleted += 1
            else:
                failed += 1
        
        status = f"Cleared {deleted} documents"
        if failed > 0:
            status += f", {failed} deletions failed"
        
        return status
        
    except Exception as e:
        return f"Error: {str(e)}"