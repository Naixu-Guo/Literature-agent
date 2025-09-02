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


def _get_early_chunks(doc_id: str, max_chunks: int = 5) -> List[str]:
    """
    Get early chunks from a document, which typically contain introduction, methods, and results.
    Avoids late chunks that usually contain references/bibliography.
    
    :param doc_id: document ID
    :param max_chunks: maximum number of early chunks to return
    :return: list of early chunk contents
    """
    try:
        doc = _agent.documents.get(doc_id, {})
        chunks = doc.get("chunks", [])
        
        if not chunks:
            return []
        
        # Get first N chunks, but limit to first 1/3 of document to avoid references
        total_chunks = len(chunks)
        early_limit = min(max_chunks, max(5, total_chunks // 3))  # At least 5, or 1/3 of doc
        
        return chunks[:early_limit]
        
    except Exception as e:
        print(f"Warning: Failed to get early chunks for {doc_id}: {e}")
        return []


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
            # Multiple documents - get content from early chunks for better integration
            summary_parts = []
            
            for doc_id in doc_ids:
                # Get early chunks (0-4) which contain intro, methods, results
                early_chunks = _get_early_chunks(doc_id, max_chunks=5)
                
                if early_chunks:
                    # Summarize each document's key content
                    doc_summary = _agent.summarize(doc_id, max_length=max_length//len(doc_ids))
                    if not doc_summary.get("error"):
                        title = _agent.documents.get(doc_id, {}).get("metadata", {}).get("title", f"Document {doc_id}")
                        summary_parts.append(f"**{title}**: {doc_summary.get('summary', '')}")
            
            if summary_parts:
                # Create integrated summary from individual summaries
                combined_content = "\n\n".join(summary_parts)
                
                # Use LLM to integrate the summaries
                llm = _agent._get_llm()
                if llm:
                    integration_prompt = f"""Integrate these {len(doc_ids)} document summaries into a cohesive summary highlighting common themes, complementary insights, and key differences. Keep under {max_length} characters.

{combined_content}

Integrated summary:"""
                    
                    try:
                        response = llm.invoke(integration_prompt)
                        summary = response.content
                    except Exception as e:
                        # Fallback to concatenated summaries
                        summary = f"Summary of {len(doc_ids)} documents:\n\n" + combined_content
                else:
                    summary = f"Summary of {len(doc_ids)} documents:\n\n" + combined_content
            else:
                summary = f"Unable to generate integrated summary for {len(doc_ids)} documents"
        
        # Add information about failed sources if any
        if failed_sources:
            summary += f"\n\nNote: {len(failed_sources)} source(s) failed to load: {'; '.join(failed_sources)}"
        
        return summary
        
    except Exception as e:
        return f"Error: {str(e)}"






@toolset.add()
def query_documents(query: str, num_results: int = 5) -> str:
    """
    Query documents and get the best answer combining AI analysis with direct quotes.
    Automatically provides comprehensive AI-generated response enhanced with key excerpts.
    
    :param query: search query or question
    :param num_results: number of results/sources to use (default: 5)
    :return: comprehensive answer with AI analysis and direct quotes
    """
    try:
        # Get search results for direct quotes
        search_result = _agent.search(query, num_results=num_results)
        excerpts = search_result.get("results", []) if "error" not in search_result else []
        
        # Get AI-generated answer
        rag_result = _agent.ask_question(query, num_sources=num_results)
        
        if "error" in rag_result:
            return f"Error: {rag_result['error']}"
        
        ai_answer = rag_result.get("answer", "No answer available")
        sources = rag_result.get("sources", [])
        
        # Build comprehensive response
        response = f"{ai_answer}\n"
        
        # Add key excerpts if available
        if excerpts:
            response += f"\n**Key Excerpts:**\n"
            for i, excerpt in enumerate(excerpts[:3], 1):  # Top 3 most relevant
                # Extract content from the dictionary result
                if isinstance(excerpt, dict):
                    content = excerpt.get("content", str(excerpt))
                    doc_title = excerpt.get("document_title", "Unknown")
                    clean_excerpt = content.strip() if isinstance(content, str) else str(content)
                else:
                    clean_excerpt = str(excerpt).strip()
                    doc_title = "Unknown"
                
                if len(clean_excerpt) > 250:
                    clean_excerpt = clean_excerpt[:250] + "..."
                response += f"{i}. \"{clean_excerpt}\" _(from {doc_title})_\n\n"
        
        # Add sources
        if sources:
            response += "**Sources:**\n"
            for i, source in enumerate(sources, 1):
                response += f"{i}. {source}\n"
        
        return response
        
    except Exception as e:
        return f"Error: {str(e)}"


def _find_arxiv_alternative(title: str, snippet: str, original_url: str) -> Optional[str]:
    """
    Try to find an arXiv PDF alternative for a paper that requires authentication.
    
    :param title: paper title
    :param snippet: paper snippet/abstract
    :param original_url: original URL that failed
    :return: arXiv PDF URL if found, None otherwise
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import quote
        import re
        
        # Extract potential arXiv ID from original URL
        arxiv_id_match = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', original_url)
        if arxiv_id_match:
            arxiv_id = arxiv_id_match.group(1)
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        # Search arXiv directly using title
        clean_title = re.sub(r'[^\w\s]', ' ', title).strip()
        search_terms = ' '.join(clean_title.split()[:8])  # Use first 8 words
        
        arxiv_search_url = f"https://arxiv.org/search/?query={quote(search_terms)}&searchtype=all&source=header"
        
        headers = {'User-Agent': 'Literature-Agent/1.0'}
        response = requests.get(arxiv_search_url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for search results
        for result in soup.find_all('li', class_='arxiv-result'):
            result_title_elem = result.find('p', class_='title')
            if not result_title_elem:
                continue
                
            result_title = result_title_elem.get_text(strip=True)
            
            # Check if titles are similar (basic similarity check)
            if _titles_similar(title, result_title):
                # Extract arXiv ID from the result
                link_elem = result.find('a', href=True)
                if link_elem and '/abs/' in link_elem['href']:
                    arxiv_id = link_elem['href'].split('/abs/')[-1]
                    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        return None
        
    except Exception as e:
        print(f"Warning: arXiv fallback search failed: {e}")
        return None


def _titles_similar(title1: str, title2: str) -> bool:
    """
    Basic similarity check between two paper titles.
    
    :param title1: first title
    :param title2: second title
    :return: True if titles are considered similar
    """
    # Normalize titles: remove punctuation, convert to lowercase, split into words
    def normalize_title(title):
        import re
        # Remove common prefixes/suffixes and punctuation
        clean = re.sub(r'[^\w\s]', ' ', title.lower())
        clean = re.sub(r'\b(the|a|an|on|in|for|with|of|to|and|or)\b', ' ', clean)
        return set(clean.split())
    
    words1 = normalize_title(title1)
    words2 = normalize_title(title2)
    
    # Calculate Jaccard similarity (intersection / union)
    if not words1 or not words2:
        return False
        
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    similarity = intersection / union if union > 0 else 0
    return similarity > 0.6  # 60% similarity threshold


@toolset.add()
def web_research(query: str, num_results: int = 5) -> str:
    """
    Search and load web documents related to a query, returning a comprehensive summary.
    Automatically finds arXiv PDF alternatives for papers behind authentication walls.
    
    :param query: search query
    :param num_results: maximum number of results to load (default: 5)
    :return: formatted results with summaries and metadata
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
        
        try:
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
        except Exception as e:
            return f"Error accessing Google Scholar: {str(e)}"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        loaded_count = 0
        failed_loads = []
        
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
            final_url = url
            final_status = "original"
            
            # If loading failed, try to find arXiv alternative
            if not load_result.get("success"):
                arxiv_url = _find_arxiv_alternative(title, snippet, url)
                if arxiv_url:
                    load_result = _agent.load_url(arxiv_url, title=title)
                    if load_result.get("success"):
                        final_url = arxiv_url
                        final_status = "arxiv_fallback"
            
            if load_result.get("success"):
                doc_id = load_result["doc_id"]
                
                # Generate brief summary (1-2 sentences) for quick overview
                summary_result = _agent.summarize(doc_id, max_length=100)
                
                # Use snippet as fallback if summary fails
                if summary_result.get("error"):
                    summary = snippet[:150] + "..." if snippet else "Document loaded successfully"
                else:
                    summary = summary_result.get("summary", "")[:200]  # Limit to 200 chars
                
                results.append({
                    "title": title,
                    "url": final_url,
                    "original_url": url if final_status == "arxiv_fallback" else None,
                    "snippet": snippet,
                    "summary": summary,
                    "doc_id": doc_id,
                    "status": final_status
                })
                loaded_count += 1
            else:
                failed_loads.append(f"{title}: {load_result.get('error', 'Unknown error')}")
        
        # Format results
        if not results and not failed_loads:
            return f"No results found for query: '{query}'. Google Scholar may be blocking requests or no accessible documents found."
        
        output = f"**Web Research Results for: '{query}'**\n\n"
        
        if results:
            output += f"**Successfully loaded {len(results)} document(s):**\n\n"
            for i, result in enumerate(results, 1):
                output += f"**{i}. {result['title']}**\n"
                if result.get('original_url'):
                    output += f"✅ arXiv: {result['url']}\n"
                    output += f"(Original: {result['original_url']})\n"
                else:
                    output += f"URL: {result['url']}\n"
                output += f"Brief: {result['summary']}\n"
                output += f"ID: {result['doc_id']}\n\n"
        
        if failed_loads:
            output += f"\n**Failed to load {len(failed_loads)} document(s):**\n"
            for failure in failed_loads[:3]:  # Show first 3 failures
                output += f"• {failure}\n"
        
        if loaded_count > 0:
            output += f"\n**Note:** {loaded_count} documents have been added to your knowledge base and can be queried using `query_documents()`."
        
        return output
        
    except Exception as e:
        return f"Error in web research: {str(e)}"




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