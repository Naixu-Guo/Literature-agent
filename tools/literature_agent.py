"""
Unified MCP-based Literature Agent
All functions use the MCP backend for optimal performance and consistency
"""

import os
import json
from typing import Optional, Union, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from tools.toolset import toolset
from tools.mcp_backend import _agent


def _auto_optimize_num_results(query: str) -> int:
    """
    Automatically optimize num_results based on query complexity and corpus size.
    
    :param query: the search query
    :return: optimized number of results
    """
    try:
        # Get corpus size
        corpus_size = len(_agent.documents)
        
        # Analyze query complexity
        query_words = len(query.split())
        has_complex_terms = any(term in query.lower() for term in [
            'compare', 'contrast', 'difference', 'relationship', 'how', 'why', 
            'explain', 'analyze', 'comprehensive', 'detailed', 'multiple', 'various'
        ])
        
        # Base calculation
        if query_words < 5:
            base_results = 3  # Simple queries
        elif query_words < 10:
            base_results = 5  # Medium queries  
        else:
            base_results = 7  # Complex queries
            
        # Adjust for complex terms
        if has_complex_terms:
            base_results += 2
            
        # Scale with corpus size
        if corpus_size > 50:
            base_results += 2
        elif corpus_size > 20:
            base_results += 1
            
        # Cap the results
        return min(max(base_results, 3), 12)  # Between 3-12
        
    except Exception:
        return 5  # Safe fallback


def _enhanced_search(query: str, num_results: int) -> dict:
    """
    Enhanced search that prioritizes content-rich chunks over references/bibliography.
    
    :param query: search query
    :param num_results: number of results to return
    :return: search results dictionary
    """
    try:
        # Get more results than needed for filtering
        expanded_results = _agent.search(query, num_results=num_results * 2)
        
        if "error" in expanded_results:
            return expanded_results
            
        all_results = expanded_results.get("results", [])
        
        # Score and filter results
        scored_results = []
        for result in all_results:
            chunk_id = result.get("chunk_id", 0)
            doc_id = result.get("doc_id", "")
            content = result.get("content", "")
            
            # Get total chunks for this document  
            doc = _agent.documents.get(doc_id, {})
            total_chunks = len(doc.get("chunks", []))
            
            # Calculate content quality score
            score = _calculate_content_score(content, chunk_id, total_chunks)
            
            scored_results.append({
                **result,
                "content_score": score
            })
        
        # Sort by content quality, then by similarity score
        scored_results.sort(key=lambda x: (x["content_score"], -x["score"]), reverse=True)
        
        # Return top results
        filtered_results = scored_results[:num_results]
        
        return {
            "query": query,
            "num_results": len(filtered_results),
            "results": filtered_results
        }
        
    except Exception as e:
        # Fallback to regular search
        return _agent.search(query, num_results)


def _calculate_content_score(content: str, chunk_id: int, total_chunks: int) -> float:
    """
    Calculate content quality score to prioritize substantive content.
    
    :param content: chunk content
    :param chunk_id: position of chunk in document
    :param total_chunks: total chunks in document
    :return: content quality score (0-1)
    """
    score = 0.0
    
    # Position scoring: early chunks are more valuable
    position_ratio = chunk_id / max(total_chunks, 1)
    if position_ratio < 0.3:  # First 30% of document
        score += 0.4
    elif position_ratio < 0.6:  # Middle 30%
        score += 0.2
    else:  # Last 40% (likely references)
        score -= 0.2
    
    # Content quality indicators
    content_lower = content.lower()
    
    # Positive indicators (substantive content)
    positive_terms = [
        'algorithm', 'method', 'approach', 'result', 'experiment', 'analysis',
        'propose', 'introduce', 'demonstrate', 'prove', 'show', 'find',
        'conclusion', 'contribution', 'novel', 'technique', 'performance',
        'implementation', 'solution', 'framework', 'model', 'theory'
    ]
    
    # Negative indicators (references/metadata)
    negative_terms = [
        'et al', 'bibliography', 'references', 'doi:', 'arxiv:', 'isbn',
        'proceedings', 'conference', 'journal', 'volume', 'pages',
        'publisher', 'citation', 'cited', 'reference'
    ]
    
    # Score based on content indicators
    positive_count = sum(1 for term in positive_terms if term in content_lower)
    negative_count = sum(1 for term in negative_terms if term in content_lower)
    
    score += min(positive_count * 0.05, 0.3)  # Max +0.3
    score -= min(negative_count * 0.1, 0.4)   # Max -0.4
    
    # Length bonus for substantial content
    if len(content) > 500:
        score += 0.1
    elif len(content) < 100:
        score -= 0.1
    
    return max(0.0, min(1.0, score))  # Clamp to 0-1


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
def query_documents(query: str, mode: str = "algorithm_spec", num_results: int = -1) -> str:
    """
    Query documents with a focus on producing a high-quality, structured description of
    quantum algorithms suited for downstream resource estimation.

    - mode="algorithm_spec" (default): returns strict JSON string per schema
    - mode="answer": returns a concise textual answer with simple source list
    """
    try:
        # Auto-optimize num_results if requested
        if num_results == -1:
            num_results = _auto_optimize_num_results(query)

        if mode == "algorithm_spec":
            result = _agent.extract_algorithm_spec(query, num_sources=num_results)
            if "error" in result:
                return f"Error: {result['error']}"
            return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

        if mode == "answer":
            rag = _agent.ask_question(query, num_sources=num_results)
            if "error" in rag:
                return f"Error: {rag['error']}"
            answer = rag.get("answer", "No answer available")
            sources = rag.get("sources", [])
            # Minimal, clean formatting
            resp = answer.strip() + "\n"
            if sources:
                resp += "\nSources:\n" + "\n".join(
                    f"- {s.get('document','Unknown')} (doc {s.get('doc_id')}, chunk {s.get('chunk_id')})" for s in sources
                )
            return resp

        return "Error: Invalid mode. Use 'algorithm_spec' or 'answer'."

    except Exception as e:
        return f"Error: {str(e)}"


def _enhanced_rag_answer(query: str, excerpts: List[dict]) -> dict:
    """
    Generate enhanced AI answer with better prompting and context awareness.
    
    :param query: the search query
    :param excerpts: search result excerpts
    :return: enhanced RAG result
    """
    try:
        llm = _agent._get_llm()
        if not llm:
            return {"error": "LLM not available. Please set GOOGLE_API_KEY."}
        
        if not excerpts:
            return {
                "answer": "No relevant information found in the document corpus.",
                "sources": []
            }
        
        # Build context with source attribution (limit context size for API safety)
        context_parts = []
        sources = []
        max_context_length = 8000  # Safe limit for API calls
        current_length = 0
        
        for i, excerpt in enumerate(excerpts):
            doc_title = excerpt.get("document_title", "Unknown Document")[:200]  # Limit title length
            content = excerpt.get("content", "")[:1000]  # Limit content length
            
            # Check if adding this excerpt would exceed the limit
            excerpt_text = f"[Source {i+1}: {doc_title}]\n{content}\n"
            if current_length + len(excerpt_text) > max_context_length:
                break
                
            context_parts.append(excerpt_text)
            sources.append(doc_title)
            current_length += len(excerpt_text)
        
        if not context_parts:
            return {
                "answer": "Content too large to process safely.",
                "sources": []
            }
        
        context = "\n".join(context_parts)
        
        # Enhanced prompt for better answer quality
        prompt = f"""Based on the provided research excerpts, answer the question with academic rigor.

Instructions:
- Provide a comprehensive, well-structured answer
- Cite specific findings and evidence from the sources
- Use academic tone but remain accessible
- If information is incomplete, clearly state limitations

Context from {len(context_parts)} research excerpts:
{context}

Question: {query[:500]}  

Answer:"""
        
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        return {"error": f"Answer generation failed: {str(e)}"}


def _curate_excerpts(excerpts: List[dict], query: str, max_excerpts: int = 3) -> List[dict]:
    """
    Select and format high-quality, diverse excerpts for display.
    
    :param excerpts: search result excerpts
    :param query: original query for relevance scoring  
    :param max_excerpts: maximum number of excerpts to return
    :return: list of curated excerpt information
    """
    if not excerpts:
        return []
    
    try:
        # Simple scoring and selection for API safety
        scored_excerpts = []
        seen_docs = set()
        
        for excerpt in excerpts:
            doc_id = excerpt.get("doc_id", "")
            doc_title = excerpt.get("document_title", "Unknown")[:100]  # Limit title length
            content = excerpt.get("content", "")[:400]  # Limit content length
            similarity_score = float(excerpt.get("score", 0.0))
            
            # Simple relevance check
            query_words = set(word.lower() for word in query.split() if len(word) > 2)
            content_words = set(word.lower() for word in content.split() if len(word) > 2)
            relevance = len(query_words.intersection(content_words)) / max(len(query_words), 1)
            
            # Diversity bonus for new documents
            diversity_bonus = 0.3 if doc_id not in seen_docs else 0.0
            seen_docs.add(doc_id)
            
            # Combined score
            combined_score = similarity_score + relevance + diversity_bonus
            
            scored_excerpts.append({
                "doc_title": doc_title,
                "content": content,
                "score": combined_score
            })
        
        # Sort and select top excerpts
        scored_excerpts.sort(key=lambda x: x["score"], reverse=True)
        selected = scored_excerpts[:max_excerpts]
        
        # Format for display with safe truncation
        curated = []
        for item in selected:
            content = item["content"]
            doc_title = item["doc_title"]
            
            # Safe truncation
            if len(content) > 250:
                content = content[:247] + "..."
            
            curated.append({
                "formatted_excerpt": f'*"{content}"* — {doc_title}'
            })
        
        return curated
        
    except Exception:
        # Simple fallback formatting
        return [{
            "formatted_excerpt": f'*"{excerpt.get("content", "")[:200]}..."* — {excerpt.get("document_title", "Unknown")[:50]}'
        } for excerpt in excerpts[:max_excerpts]]


def _clean_sources(sources: List[str]) -> List[str]:
    """
    Clean and deduplicate source citations for better presentation.
    
    :param sources: list of source strings
    :return: cleaned and deduplicated source list
    """
    if not sources:
        return []
    
    try:
        # Simple deduplication and cleaning
        seen = set()
        cleaned = []
        
        for source in sources:
            if not source or not source.strip():
                continue
                
            # Basic cleaning - limit length and remove IDs
            clean_source = source.strip()[:150]  # Limit length for API safety
            
            # Remove ID patterns
            if " (ID:" in clean_source:
                clean_source = clean_source.split(" (ID:")[0]
            
            # Avoid duplicates
            source_key = clean_source.lower()
            if source_key not in seen and len(clean_source) > 5:
                seen.add(source_key)
                cleaned.append(clean_source)
        
        return cleaned[:10]  # Limit to 10 sources for API safety
        
    except Exception:
        # Simple fallback
        return list(set(str(s)[:100] for s in sources if s))[:5]


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