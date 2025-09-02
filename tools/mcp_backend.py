"""
Simple MCP Integration for Literature Agent
Provides MCP functionality without complex async handling
"""

from typing import List, Optional, Dict, Any
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
        self.vector_store_dir = self.storage_dir / "vector_stores"
        self.vector_store_dir.mkdir(exist_ok=True)
        
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
        self._load_vector_stores()
    
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
        except Exception:
            return None
    
    def _get_embeddings(self):
        """Get embeddings instance with fresh API key and robust error handling"""
        api_key = self._get_api_key()
        if not api_key or api_key.startswith("your-"):
            return None
        
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model=os.getenv("EMBED_MODEL", "models/embedding-001"),
                google_api_key=api_key,
            )
            
            # Test embeddings with a simple query to ensure they work
            try:
                embeddings.embed_query("test")
                return embeddings
            except Exception as test_error:
                print(f"Embeddings test failed: {test_error}")
                return None
                
        except Exception as creation_error:
            print(f"Embeddings creation failed: {creation_error}")
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
                self._save_vector_store(doc_id)
                has_embeddings = True
            except Exception:
                pass
        
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
        """Search across documents with improved ranking for research papers"""
        embeddings = self._get_embeddings()
        if not embeddings:
            return {"error": "Embeddings not available. Please set GOOGLE_API_KEY."}
        
        try:
            search_doc_ids = doc_ids if doc_ids else list(self.documents.keys())
            
            if not search_doc_ids:
                return {"error": "No documents loaded"}
            
            all_results = []
            query_lower = query.lower()
            
            # Categorize documents by relevance first
            high_priority_docs = []
            medium_priority_docs = []
            low_priority_docs = []
            
            for doc_id in search_doc_ids:
                if doc_id not in self.vector_stores:
                    continue
                    
                doc_info = self.documents.get(doc_id, {})
                doc_title = doc_info.get("metadata", {}).get("title", "").lower()
                doc_source = doc_info.get("source", "").lower()
                
                # Skip test/dummy documents
                if any(skip_term in doc_title or skip_term in doc_source for skip_term in [
                    "httpbin.org", "dummy.pdf", "test_doc", "uuid", "hello world", "/tmp"
                ]):
                    continue
                
                # High priority: exact keyword matches in title
                query_keywords = query_lower.split()
                title_keyword_matches = sum(1 for kw in query_keywords if kw in doc_title)
                
                if title_keyword_matches >= 2 or any(term in doc_title for term in [
                    "singular value transformation", "qsvt", "nonlinear transformation"
                ]):
                    high_priority_docs.append(doc_id)
                elif any(term in doc_title for term in ["quantum", "algorithm"]):
                    medium_priority_docs.append(doc_id)
                else:
                    low_priority_docs.append(doc_id)
            
            # Search prioritized document groups
            for priority_group, boost in [(high_priority_docs, 0.8), (medium_priority_docs, 0.3), (low_priority_docs, 0.0)]:
                for doc_id in priority_group:
                    vector_store = self.vector_stores[doc_id]
                    results = vector_store.similarity_search_with_score(query, k=min(num_results, 5))
                    
                    for doc, score in results:
                        # Apply priority boost
                        adjusted_score = float(score) - boost
                        
                        all_results.append({
                            "doc_id": doc_id,
                            "document_title": self.documents[doc_id]["metadata"].get("title", doc_id),
                            "chunk_id": doc.metadata.get("chunk_id", 0),
                            "content": doc.page_content,
                            "score": adjusted_score
                        })
            
            # Sort by adjusted score and return top results
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

    def extract_algorithm_spec(self, query: str, doc_ids: Optional[List[str]] = None, num_sources: int = 5) -> Dict[str, Any]:
        """Extract a structured quantum algorithm specification suitable for resource estimation.

        Returns a dict with keys: algorithm_spec (object), sources (list), and schema_version.
        """
        llm = self._get_llm()
        if not llm:
            return {"error": "LLM not available. Please set GOOGLE_API_KEY."}

        try:
            search_result = self.search(query, num_sources, doc_ids)
            if "error" in search_result:
                return search_result

            results = search_result.get("results", [])
            if not results:
                return {"error": "No relevant context found to extract algorithm specification"}

            context_parts: List[str] = []
            sources: List[Dict[str, Any]] = []
            for res in results:
                # Use full chunk content for maximum quality
                content = res['content']
                doc_title = res['document_title'][:50]  # Shorten title for context
                context_parts.append(f"[{doc_title}]\n{content}\n")
                sources.append({
                    "document": res["document_title"],
                    "doc_id": res["doc_id"],
                    "chunk_id": res["chunk_id"]
                })

            context = "\n".join(context_parts)

            schema = {
                "algorithm_spec": {
                    "algorithm_name": None,
                    "problem_description": None,
                    "input_parameters": {},
                    "output": None,
                    "computational_model": {
                        "gate_set": None,
                        "oracle_model": None,
                        "precision": None,
                        "error_correction": None
                    },
                    "resources": {
                        "logical_qubits": None,
                        "ancilla_qubits": None,
                        "t_count": None,
                        "t_depth": None,
                        "toffoli_count": None,
                        "cnot_count": None,
                        "total_gates": None,
                        "circuit_depth": None,
                        "runtime_complexity": None,
                        "space_complexity": None,
                        "query_complexity": None
                    },
                    "subroutines": [],
                    "algorithm_steps": [],
                    "key_techniques": [],
                    "assumptions": [],
                    "success_probability": None,
                    "error_bounds": None,
                    "applications": []
                }
            }

            instruction = (
                "You are a quantum computing expert extracting comprehensive algorithm specifications for resource estimation. "
                "Analyze the provided research paper context to extract ALL available information about quantum algorithms. "
                "IMPORTANT: Extract information even if it requires inference from technical descriptions - don't leave fields null if you can reasonably infer values. "
                "For resource counts: include both explicit numbers AND reasonable estimates based on algorithm descriptions. "
                "For example: if text mentions 'n qubits for encoding' or 'O(n) ancilla qubits', extract this information. "
                "For algorithm steps: break down the algorithm into clear, numbered steps based on the methodology described. "
                "For techniques: identify quantum techniques like QSVT, amplitude amplification, quantum walks, etc. "
                "For complexity: extract or infer time/space complexity from algorithmic analysis in the paper. "
                "Return detailed, actionable information suitable for quantum resource estimation. "
                "Return pure JSON only, no markdown."
            )

            # Use maximum context for quality (up to 15k chars for better coverage)
            max_context = min(len(context), 15000)
            prompt = f"""{instruction}

Extract comprehensive algorithm details from the context below.

SCHEMA:
{json.dumps(schema, indent=2)}

CONTEXT:
{context[:max_context]}

QUERY: {query}

OUTPUT (JSON only):"""

            response = llm.invoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)

            # Try to parse JSON strictly
            try:
                data = json.loads(raw)
            except Exception:
                # Attempt to salvage JSON by finding first/last braces
                start = raw.find('{')
                end = raw.rfind('}')
                if start != -1 and end != -1 and end > start:
                    try:
                        data = json.loads(raw[start:end+1])
                    except Exception as parse_err:
                        return {"error": f"Model did not return valid JSON: {parse_err}"}
                else:
                    return {"error": "Model did not return JSON"}

            # Minimal validation
            if not isinstance(data, dict) or "algorithm_spec" not in data:
                return {"error": "JSON missing 'algorithm_spec' object"}

            # Attach sources if not present
            data.setdefault("sources", sources)
            data.setdefault("schema_version", "1.0")
            return data

        except Exception as e:
            return {"error": f"Algorithm spec extraction failed: {str(e)}"}
    
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
        # Save all document data including content for full functionality
        with open(doc_file, 'w') as f:
            json.dump(doc_info, f, indent=2)
    
    def _load_saved_documents(self):
        """Load saved documents from disk"""
        for doc_file in self.storage_dir.glob("*.json"):
            try:
                with open(doc_file, 'r') as f:
                    data = json.load(f)
                    self.documents[data["id"]] = data
            except Exception:
                pass
    
    def _save_vector_store(self, doc_id: str):
        """Save vector store to disk"""
        if doc_id not in self.vector_stores:
            return
        
        try:
            vector_store_file = self.vector_store_dir / f"{doc_id}.faiss"
            self.vector_stores[doc_id].save_local(str(vector_store_file))
        except Exception:
            pass
    
    def _load_vector_stores(self):
        """Load vector stores from disk with robust error handling"""
        embeddings = self._get_embeddings()
        if not embeddings:
            return
        
        try:
            from langchain_community.vectorstores import FAISS
            loaded_count = 0
            failed_count = 0
            
            for vector_store_dir in self.vector_store_dir.iterdir():
                if vector_store_dir.is_dir():
                    doc_id = vector_store_dir.name.replace('.faiss', '')
                    if doc_id in self.documents:
                        try:
                            vector_store = FAISS.load_local(str(vector_store_dir), embeddings, allow_dangerous_deserialization=True)
                            self.vector_stores[doc_id] = vector_store
                            loaded_count += 1
                        except Exception as e:
                            print(f"Failed to load vector store for {doc_id}: {e}")
                            failed_count += 1
            
            if loaded_count > 0:
                print(f"Loaded {loaded_count} vector stores, {failed_count} failed")
        except Exception as e:
            print(f"Vector store loading failed: {e}")
    
    def check_embeddings_health(self) -> dict:
        """Check embeddings health and provide recovery options"""
        try:
            embeddings = self._get_embeddings()
            if not embeddings:
                return {"healthy": False, "issue": "Embeddings API not available", "fix": "Check GOOGLE_API_KEY"}
            
            docs_with_content = sum(1 for doc in self.documents.values() if doc.get("chunks"))
            docs_with_embeddings = len(self.vector_stores)
            
            health_status = {
                "healthy": docs_with_embeddings == docs_with_content,
                "total_documents": len(self.documents),
                "documents_with_content": docs_with_content,
                "documents_with_embeddings": docs_with_embeddings,
                "missing_embeddings": docs_with_content - docs_with_embeddings
            }
            
            if not health_status["healthy"]:
                health_status["fix"] = "Run create_embeddings('all') to fix missing embeddings"
            
            return health_status
            
        except Exception as e:
            return {"healthy": False, "error": str(e)}


# Global instance
_agent = SimpleMCPLiteratureAgent()

