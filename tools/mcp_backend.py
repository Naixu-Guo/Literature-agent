"""
Simple MCP Integration for Literature Agent
Provides MCP functionality without complex async handling
"""

from typing import List, Optional
import os
import json
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
import ipaddress
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup


class SimpleMCPLiteratureAgent:
    """Simple MCP-style Literature Agent without complex async handling"""
    
    def __init__(self):
        self.documents = {}
        self.vector_stores = {}
        self.storage_dir = Path("mcp_literature_storage")
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize LLM and embeddings as None - will be created dynamically
        self.llm = None
        self.embeddings = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self._load_saved_documents()
    
    def _get_api_key(self):
        """Get fresh API key by reloading .env file"""
        load_dotenv(override=True)  # Reload .env file
        return os.getenv("GOOGLE_API_KEY")
    
    def _get_llm(self):
        """Get LLM instance with fresh API key"""
        api_key = self._get_api_key()
        if not api_key or api_key.startswith("your-"):
            return None
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
                google_api_key=api_key,
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            )
        except Exception as e:
            print(f"Warning: Could not create LLM: {e}")
            return None
    
    def _get_embeddings(self):
        """Get embeddings instance with fresh API key"""
        api_key = self._get_api_key()
        if not api_key or api_key.startswith("your-"):
            return None
        
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model=os.getenv("EMBED_MODEL", "models/embedding-001"),
                google_api_key=api_key,
            )
        except Exception as e:
            print(f"Warning: Could not create embeddings: {e}")
            return None
    
    def load_file(self, path: str, title: Optional[str] = None) -> dict:
        """Load and process a file (PDF or text)"""
        try:
            file_path = Path(path)
            if not file_path.exists():
                return {"error": f"File not found: {path}"}
            
            # Handle PDF files
            if path.lower().endswith('.pdf'):
                with open(file_path, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() or ""
                
                metadata = {
                    "title": title or file_path.stem,
                    "source_type": "pdf",
                    "path": str(file_path),
                    "pages": len(pdf_reader.pages),
                    "file_size": file_path.stat().st_size
                }
            else:
                # Handle text files
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                
                metadata = {
                    "title": title or file_path.stem,
                    "source_type": "text",
                    "path": str(file_path),
                    "file_size": file_path.stat().st_size
                }
            
            doc_id = hashlib.md5(f"{path}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
            
            return self._process_document(doc_id, str(file_path), text, metadata)
            
        except Exception as e:
            return {"error": f"Failed to load file: {str(e)}"}
    
    def load_pdf(self, path: str, title: Optional[str] = None) -> dict:
        """Load and process a PDF document (backward compatibility)"""
        return self.load_file(path, title)
    
    def load_url(self, url: str, title: Optional[str] = None) -> dict:
        """Load and process content from a URL with basic SSRF and size protections"""
        try:
            # Basic SSRF protection: block private/internal IPs and file schemes
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"}:
                return {"error": "Only http/https URLs are allowed"}

            # Disallow obvious localhost/internal hosts
            hostname = (parsed.hostname or "").lower()
            blocked_hosts = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "169.254.169.254"}
            if hostname in blocked_hosts:
                return {"error": "Access to local/loopback addresses is not allowed"}

            # Block direct private/reserved IP access
            try:
                ip = ipaddress.ip_address(hostname)
                if (
                    ip.is_private
                    or ip.is_loopback
                    or ip.is_link_local
                    or ip.is_reserved
                    or ip.is_multicast
                ):
                    return {"error": "Access to private/reserved IP ranges is not allowed"}
            except ValueError:
                # Not an IP literal; proceed
                pass

            # Enforce content length limit via headers when available
            max_bytes = int(os.getenv("MAX_DOWNLOAD_BYTES", "10485760"))  # 10 MB default
            headers = {"User-Agent": os.getenv("USER_AGENT", "Literature-Agent/1.0")}
            with requests.get(url, headers=headers, stream=True, timeout=30) as response:
                response.raise_for_status()
                # Re-validate after redirects
                final_host = urlparse(response.url).hostname or ""
                final_host_l = final_host.lower()
                if final_host_l in blocked_hosts:
                    return {"error": "Redirected to a disallowed host"}
                try:
                    final_ip = ipaddress.ip_address(final_host_l)
                    if (
                        final_ip.is_private
                        or final_ip.is_loopback
                        or final_ip.is_link_local
                        or final_ip.is_reserved
                        or final_ip.is_multicast
                    ):
                        return {"error": "Redirected to a private/reserved IP"}
                except ValueError:
                    pass
                content_type = response.headers.get("Content-Type", "")
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > max_bytes:
                    return {"error": "Remote file too large"}

                if url.endswith('.pdf') or 'application/pdf' in content_type:
                    # Handle PDF URLs with streamed write and size cap
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=65536):
                            if not chunk:
                                break
                            downloaded += len(chunk)
                            if downloaded > max_bytes:
                                tmp_path = tmp_file.name
                                tmp_file.close()
                                Path(tmp_path).unlink(missing_ok=True)
                                return {"error": "Downloaded file exceeded size limit"}
                            tmp_file.write(chunk)
                        tmp_path = tmp_file.name
                    try:
                        result = self.load_pdf(tmp_path, title or url)
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
                    return result
                else:
                    # Handle HTML safely with size cap
                    text_chunks: List[str] = []
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=65536, decode_unicode=True):
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            return {"error": "Downloaded content exceeded size limit"}
                        text_chunks.append(chunk)
                    html_text = ''.join(text_chunks)

            soup = BeautifulSoup(html_text, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            doc_id = hashlib.md5(f"{url}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
            
            metadata = {
                "title": title or (soup.title.string if getattr(soup, 'title', None) and soup.title.string else url),
                "source_type": "url",
                "url": url,
                "content_length": len(text)
            }
            
            return self._process_document(doc_id, url, text, metadata)
                
        except Exception as e:
            return {"error": f"Failed to load URL: {str(e)}"}
    
    def _process_document(self, doc_id: str, source: str, text: str, metadata: dict) -> dict:
        """Common document processing logic"""
        # Store document info
        doc_info = {
            "id": doc_id,
            "source": source,
            "content": text,
            "metadata": metadata,
            "chunks": [],
            "summary": None,
            "created_at": datetime.now().isoformat()
        }
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        doc_info["chunks"] = chunks
        
        # Create embeddings if available
        has_embeddings = False
        embeddings = self._get_embeddings()
        if embeddings:
            try:
                documents = [
                    Document(
                        page_content=chunk,
                        metadata={"doc_id": doc_id, "chunk_id": i},
                    )
                    for i, chunk in enumerate(chunks)
                ]

                # Import FAISS vector store lazily; handle environments without faiss-cpu
                try:
                    from langchain_community.vectorstores import FAISS  # type: ignore
                except Exception as import_error:
                    raise RuntimeError(
                        "FAISS vector store is unavailable. Install 'faiss-cpu' to enable embeddings."
                    ) from import_error

                vector_store = FAISS.from_documents(documents, embeddings)
                self.vector_stores[doc_id] = vector_store
                has_embeddings = True
            except Exception as e:
                print(f"Warning: Failed to create embeddings: {e}")
        
        # Store document
        self.documents[doc_id] = doc_info
        
        # Save to disk
        self._save_document(doc_info)
        
        return {
            "success": True,
            "doc_id": doc_id,
            "title": metadata["title"],
            "source": source,
            "chunks": len(chunks),
            "has_embeddings": has_embeddings
        }
    
    def search(self, query: str, num_results: int = 5, doc_ids: Optional[List[str]] = None) -> dict:
        """Search across documents"""
        embeddings = self._get_embeddings()
        if not embeddings:
            return {"error": "Embeddings not available. Please set GOOGLE_API_KEY."}
        
        try:
            search_doc_ids = doc_ids if doc_ids else list(self.documents.keys())
            
            if not search_doc_ids:
                return {"error": "No documents loaded"}
            
            all_results = []
            
            for doc_id in search_doc_ids:
                if doc_id in self.vector_stores:
                    vector_store = self.vector_stores[doc_id]
                    results = vector_store.similarity_search_with_score(query, k=num_results)
                    
                    for doc, score in results:
                        all_results.append({
                            "doc_id": doc_id,
                            "document_title": self.documents[doc_id]["metadata"].get("title", doc_id),
                            "chunk_id": doc.metadata.get("chunk_id", 0),
                            "content": doc.page_content,
                            "score": float(score)
                        })
            
            all_results.sort(key=lambda x: x["score"])
            all_results = all_results[:num_results]
            
            return {
                "query": query,
                "num_results": len(all_results),
                "results": all_results
            }
            
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}
    
    def summarize(self, doc_id: str, max_length: int = 500) -> dict:
        """Generate summary of a document"""
        if doc_id not in self.documents:
            return {"error": f"Document {doc_id} not found"}
        
        llm = self._get_llm()
        if not llm:
            return {"error": "LLM not available. Please set GOOGLE_API_KEY."}
        
        try:
            doc = self.documents[doc_id]
            
            if doc["summary"]:
                return {
                    "doc_id": doc_id,
                    "title": doc["metadata"].get("title", doc_id),
                    "summary": doc["summary"],
                    "cached": True
                }
            
            prompt = f"""Summarize the following document in approximately {max_length} words.
            Focus on the main ideas, key findings, and important details.
            
            Document: {doc['content'][:10000]}
            
            Summary:"""
            
            response = llm.invoke(prompt)
            summary = response.content
            
            doc["summary"] = summary
            self._save_document(doc)
            
            return {
                "doc_id": doc_id,
                "title": doc["metadata"].get("title", doc_id),
                "summary": summary,
                "cached": False
            }
            
        except Exception as e:
            return {"error": f"Summarization failed: {str(e)}"}
    
    def ask_question(self, question: str, doc_ids: Optional[List[str]] = None, num_sources: int = 4) -> dict:
        """Ask a question using RAG"""
        llm = self._get_llm()
        if not llm:
            return {"error": "LLM not available. Please set GOOGLE_API_KEY."}
        
        try:
            search_result = self.search(question, num_sources, doc_ids)
            
            if "error" in search_result:
                return search_result
            
            context_parts = []
            sources = []
            
            for result in search_result["results"]:
                context_parts.append(f"[Source: {result['document_title']}]\n{result['content']}\n")
                sources.append({
                    "document": result["document_title"],
                    "doc_id": result["doc_id"],
                    "chunk_id": result["chunk_id"]
                })
            
            context = "\n".join(context_parts)
            
            prompt = f"""Based on the following context, answer the question.
            If the answer cannot be found in the context, say so.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
            
            response = llm.invoke(prompt)
            answer = response.content
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "num_sources_used": len(sources)
            }
            
        except Exception as e:
            return {"error": f"Question answering failed: {str(e)}"}
    
    def list_documents(self) -> dict:
        """List all documents"""
        return {
            "num_documents": len(self.documents),
            "documents": [
                {
                    "id": doc_id,
                    "source": doc["source"],
                    "metadata": doc["metadata"],
                    "num_chunks": len(doc["chunks"]),
                    "has_embeddings": doc_id in self.vector_stores,
                    "has_summary": doc["summary"] is not None,
                    "created_at": doc["created_at"]
                }
                for doc_id, doc in self.documents.items()
            ]
        }
    
    def delete_document(self, doc_id: str) -> dict:
        """Delete a document"""
        if doc_id not in self.documents:
            return {"error": f"Document {doc_id} not found"}
        
        try:
            del self.documents[doc_id]
            
            if doc_id in self.vector_stores:
                del self.vector_stores[doc_id]
            
            doc_file = self.storage_dir / f"{doc_id}.json"
            if doc_file.exists():
                doc_file.unlink()
            
            return {
                "success": True,
                "doc_id": doc_id,
                "message": f"Document {doc_id} deleted successfully"
            }
            
        except Exception as e:
            return {"error": f"Failed to delete document: {str(e)}"}
    
    def _save_document(self, doc_info: dict):
        """Save document to disk"""
        doc_file = self.storage_dir / f"{doc_info['id']}.json"
        save_data = {k: v for k, v in doc_info.items() if k != "content"}  # Don't save full content
        with open(doc_file, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def _load_saved_documents(self):
        """Load saved documents from disk"""
        for doc_file in self.storage_dir.glob("*.json"):
            try:
                with open(doc_file, 'r') as f:
                    data = json.load(f)
                    self.documents[data["id"]] = data
            except Exception as e:
                print(f"Failed to load document {doc_file}: {e}")


# Global instance
_agent = SimpleMCPLiteratureAgent()


def mcp_load_document(source: str, title: Optional[str] = None) -> str:
    """Load a document (PDF, text file, or URL) into the MCP system for processing."""
    
    try:
        if source.startswith("http://") or source.startswith("https://"):
            result = _agent.load_url(source, title)
        else:
            result = _agent.load_file(source, title)  # Now handles both PDF and text files
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if result.get("success"):
            return (f"✓ Document loaded successfully!\n"
                   f"  Document ID: {result.get('doc_id')}\n"
                   f"  Title: {result.get('title')}\n"
                   f"  Chunks: {result.get('chunks')}\n"
                   f"  Has embeddings: {result.get('has_embeddings', False)}")
        else:
            return f"Failed to load document: {json.dumps(result, indent=2)}"
    
    except Exception as e:
        return f"Error loading document: {str(e)}"


def mcp_search_documents(query: str, num_results: int = 5) -> str:
    """Search across all loaded documents using semantic search."""
    
    try:
        result = _agent.search(query, num_results)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        output = [f"Search Results for: '{query}'"]
        output.append(f"Found {result.get('num_results', 0)} relevant chunks\n")
        
        for i, res in enumerate(result.get("results", []), 1):
            output.append(f"Result {i}:")
            output.append(f"  Document: {res.get('document_title')}")
            output.append(f"  Score: {res.get('score', 'N/A'):.4f}")
            output.append(f"  Content: {res.get('content', '')[:200]}...")
            output.append("")
        
        return "\n".join(output)
    
    except Exception as e:
        return f"Error searching documents: {str(e)}"


def mcp_summarize_document(doc_id: str, max_length: int = 500) -> str:
    """Generate a summary of a loaded document."""
    
    try:
        result = _agent.summarize(doc_id, max_length)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        output = [f"Document Summary"]
        output.append(f"Title: {result.get('title', 'Unknown')}")
        output.append(f"Document ID: {result.get('doc_id', doc_id)}")
        output.append(f"Cached: {result.get('cached', False)}")
        output.append(f"\nSummary:\n{result.get('summary', 'No summary available')}")
        
        return "\n".join(output)
    
    except Exception as e:
        return f"Error summarizing document: {str(e)}"


def mcp_ask_question(question: str, num_sources: int = 4) -> str:
    """Ask a question about the loaded documents using RAG."""
    
    try:
        result = _agent.ask_question(question, num_sources=num_sources)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        output = [f"Question: {question}\n"]
        output.append(f"Answer:\n{result.get('answer', 'No answer available')}\n")
        
        if result.get("sources"):
            output.append("Sources:")
            for i, source in enumerate(result["sources"], 1):
                output.append(f"  {i}. {source.get('document', 'Unknown')} (chunk {source.get('chunk_id', 'N/A')})")
        
        return "\n".join(output)
    
    except Exception as e:
        return f"Error answering question: {str(e)}"


def mcp_list_documents() -> str:
    """List all documents currently loaded in the MCP system."""
    
    try:
        result = _agent.list_documents()
        
        num_docs = result.get("num_documents", 0)
        
        if num_docs == 0:
            return "No documents loaded. Use mcp_load_document() to load PDFs or URLs."
        
        output = [f"Loaded Documents ({num_docs} total):\n"]
        
        for doc in result.get("documents", []):
            output.append(f"• {doc.get('metadata', {}).get('title', 'Untitled')}")
            output.append(f"  ID: {doc.get('id')}")
            output.append(f"  Source: {doc.get('source')}")
            output.append(f"  Type: {doc.get('metadata', {}).get('source_type')}")
            output.append(f"  Chunks: {doc.get('num_chunks', 0)}")
            output.append(f"  Has embeddings: {doc.get('has_embeddings', False)}")
            output.append(f"  Has summary: {doc.get('has_summary', False)}")
            output.append("")
        
        return "\n".join(output)
    
    except Exception as e:
        return f"Error listing documents: {str(e)}"


def mcp_delete_document(doc_id: str) -> str:
    """Delete a document from the MCP system."""
    
    try:
        result = _agent.delete_document(doc_id)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if result.get("success"):
            return f"✓ {result.get('message', 'Document deleted successfully')}"
        else:
            return f"Failed to delete document: {json.dumps(result, indent=2)}"
    
    except Exception as e:
        return f"Error deleting document: {str(e)}"


def mcp_batch_load(sources: List[str]) -> str:
    """Load multiple documents at once."""
    
    if not sources:
        return "Error: No sources provided"
    
    results = []
    successful = 0
    failed = 0
    
    for source in sources:
        try:
            if source.startswith("http://") or source.startswith("https://"):
                result = _agent.load_url(source)
            else:
                result = _agent.load_pdf(source)
            
            if "error" in result:
                results.append(f"✗ {source}: {result['error']}")
                failed += 1
            elif result.get("success"):
                results.append(f"✓ {source}: Loaded as {result.get('doc_id')}")
                successful += 1
            else:
                results.append(f"✗ {source}: Unknown error")
                failed += 1
        
        except Exception as e:
            results.append(f"✗ {source}: {str(e)}")
            failed += 1
    
    output = [f"Batch Load Results: {successful} successful, {failed} failed\n"]
    output.extend(results)
    
    return "\n".join(output)