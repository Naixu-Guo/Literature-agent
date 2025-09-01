"""
Unified MCP-based Literature Agent
All functions use the MCP backend for optimal performance and consistency
"""

import os
from typing import Optional, Union, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from tools.toolset import toolset
from tools.mcp_backend import _agent


@toolset.add()
def summarize_source(sources: Union[str, List[str]], title: Optional[str] = None, max_length: int = 800) -> str:
    """
    Load and summarize content from one or multiple sources using MCP backend.
    For multiple sources, provides an integrated summary across all documents.
    Documents are automatically stored for future operations.
    
    :param sources: single source (str) or list of sources - file paths (.pdf, .txt) or URLs
    :param title: optional title for single document (ignored for multiple sources)
    :param max_length: maximum length of the summary (default: 800)
    :return: integrated summary string
    """
    try:
        # Handle single source case
        if isinstance(sources, str):
            sources = [sources]
        
        doc_ids = []
        failed_sources = []
        
        # Load all sources
        for source in sources:
            # Convert arXiv abstract URLs to PDF URLs for better loading
            if 'arxiv.org/abs/' in source:
                source = source.replace('/abs/', '/pdf/') + '.pdf'
            elif 'journals.aps.org/' in source and '/abstract/' in source:
                source = source.replace('/abstract/', '/pdf/')
            
            # Load document using MCP backend
            if source.startswith(('http://', 'https://')):
                load_result = _agent.load_url(source, title=title if len(sources) == 1 else None)
            else:
                if not os.path.isfile(source):
                    failed_sources.append(f"File not found: {source}")
                    continue
                load_result = _agent.load_file(source, title=title if len(sources) == 1 else None)
            
            if "error" in load_result or not load_result.get("success"):
                failed_sources.append(f"Failed to load: {source}")
                continue
            
            doc_ids.append(load_result["doc_id"])
        
        if not doc_ids:
            return f"No documents could be loaded. Errors: {'; '.join(failed_sources)}"
        
        # Generate integrated summary
        if len(doc_ids) == 1:
            # Single document - direct summarization
            summary_result = _agent.summarize(doc_ids[0], max_length=max_length)
            
            if "error" in summary_result:
                return f"Error generating summary: {summary_result['error']}"
            
            summary = summary_result.get("summary", "No summary available")
        else:
            # Multiple documents - use RAG for integrated summary limited to these docs
            integrated_question = (
                f"Provide a comprehensive integrated summary of the main themes, findings, and insights across all {len(doc_ids)} documents. "
                f"Maximum length: {max_length} characters."
            )
            combined_result = _agent.ask_question(integrated_question, doc_ids=doc_ids, num_sources=len(doc_ids))
            
            if "error" in combined_result:
                return f"Error generating integrated summary: {combined_result['error']}"
            
            summary = combined_result.get("answer", "No integrated summary available")
        
        # Add information about failed sources if any
        if failed_sources:
            summary += f"\n\nNote: {len(failed_sources)} source(s) failed to load: {'; '.join(failed_sources)}"
        
        return summary
        
    except Exception as e:
        return f"Error: {str(e)}"






@toolset.add()
def query_documents(query: str, mode: str = "answer", num_results: int = 5) -> str:
    """
    Query documents with two modes: search for excerpts or get AI-generated answers.
    
    :param query: search query or question
    :param mode: "search" for document excerpts, "answer" for AI-generated response (default: "answer")
    :param num_results: number of results/sources to use (default: 5)
    :return: search results or answer with sources
    """
    try:
        if mode not in ["search", "answer"]:
            return f"Invalid mode '{mode}'. Use 'search' or 'answer'."
        
        if mode == "search":
            # Search mode: return document excerpts
            result = _agent.search(query, num_results=num_results)
            
            if "error" in result:
                return f"Error: {result['error']}"
            
            matches = result.get("results", [])
            
            if not matches:
                return "No matching documents found."
            
            # Format search results
            output = f"Found {len(matches)} relevant documents:\n\n"
            for i, match in enumerate(matches, 1):
                output += f"{i}. {match}\n\n"
            
            return output
        
        else:
            # Answer mode: use RAG to generate answer
            result = _agent.ask_question(query, num_sources=num_results)
            
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
            title = doc.get('metadata', {}).get('title') or doc.get('id')
            created_at = doc.get('created_at', 'unknown')
            output += f"ID: {doc['id']}\n"
            output += f"Title: {title}\n"
            output += f"Source: {doc.get('source', 'unknown')}\n"
            output += f"Created: {created_at}\n"
            output += f"Chunks: {doc.get('num_chunks', 0)}\n"
            output += f"Has embeddings: {doc.get('has_embeddings', False)}\n"
            output += f"Has summary: {doc.get('has_summary', False)}\n"
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