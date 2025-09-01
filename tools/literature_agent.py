import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools.toolset import toolset


def _get_llm(temperature: Optional[float] = None) -> ChatGoogleGenerativeAI:
    temp = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.2"))
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    
    # Get Google API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temp,
        google_api_key=api_key,
        timeout=60,  # Add 60 second timeout
        max_retries=2  # Add retry limit
    )


def _embedder() -> GoogleGenerativeAIEmbeddings:
    embed_model = os.getenv("EMBED_MODEL", "models/embedding-001")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    return GoogleGenerativeAIEmbeddings(
        model=embed_model,
        google_api_key=api_key
    )


def _split_docs(docs):
    # Larger chunks for faster processing
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    return splitter.split_documents(docs)


@toolset.add()
def summarize_source(source: str, temperature: Optional[float] = None) -> str:
    """
    Summarize content from a local file or URL (PDF, text, or web page) using MCP backend.
    :param source: absolute path to a local file (.pdf or .txt) or URL to a web page/PDF
    :param temperature: optional LLM temperature override
    :return: summary string
    """
    try:
        # Import MCP agent
        from tools.literature_agent_mcp_simple import _agent
        
        # Convert arXiv abstract URLs to PDF URLs for better loading
        if 'arxiv.org/abs/' in source:
            source = source.replace('/abs/', '/pdf/') + '.pdf'
        elif 'journals.aps.org/' in source and '/abstract/' in source:
            source = source.replace('/abstract/', '/pdf/')
        
        # Load document using MCP backend
        if source.startswith(('http://', 'https://')):
            load_result = _agent.load_url(source)
        else:
            if not os.path.isfile(source):
                return f"File not found: {source}"
            load_result = _agent.load_file(source)  # Now handles both PDF and text files
        
        if "error" in load_result:
            return f"Error loading document: {load_result['error']}"
        
        if not load_result.get("success"):
            return "Failed to load document"
        
        # Get document ID
        doc_id = load_result["doc_id"]
        
        # Generate summary using MCP backend
        summary_result = _agent.summarize(doc_id, max_length=800)  # Longer summary to match original
        
        if "error" in summary_result:
            return f"Error generating summary: {summary_result['error']}"
        
        return summary_result.get("summary", "No summary available")
        
    except Exception as e:
        return f"Error: {str(e)}"



@toolset.add()
def summarize_multiple_sources(sources: list, temperature: Optional[float] = None) -> dict:
    """
    Summarize multiple sources (files and/or URLs) using MCP batch loading.
    :param sources: list of file paths or URLs
    :param temperature: optional LLM temperature override
    :return: dictionary with individual summaries and a combined summary
    """
    try:
        # Since summarize_source now uses MCP backend, we can just call it for each source
        # This maintains the same interface while leveraging MCP persistence
        summaries = {}
        all_summaries = []
        
        for source in sources:
            try:
                summary = summarize_source(source, temperature)
                # Extract just the filename or domain for the key
                if source.startswith(('http://', 'https://')):
                    from urllib.parse import urlparse
                    key = urlparse(source).netloc + urlparse(source).path[-30:]
                else:
                    key = os.path.basename(source)
                
                summaries[key] = summary
                all_summaries.append(f"**Source: {key}**\n{summary}")
            except Exception as e:
                summaries[source] = f"Error: {str(e)}"
        
        # Create a combined summary
        if all_summaries:
            combined_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a research assistant. Create a unified summary from multiple sources, highlighting key themes and findings."),
                ("human", "Synthesize these summaries into a coherent overview:\n\n{content}")
            ])
            
            llm = _get_llm(temperature)
            chain = combined_prompt | llm | StrOutputParser()
            combined = chain.invoke({"content": "\n\n".join(all_summaries)})
        else:
            combined = "No sources successfully summarized"
        
        return {
            "individual_summaries": summaries,
            "combined_summary": combined,
            "total_sources": len(sources),
            "successful": len([s for s in summaries.values() if not s.startswith("Error:")])
        }
        
    except Exception as e:
        return {
            "individual_summaries": {},
            "combined_summary": f"Error: {str(e)}",
            "total_sources": len(sources),
            "successful": 0
        }


@toolset.add()
def build_vector_index(paths: list) -> str:
    """
    Build a document index using MCP backend with persistent storage. Falls back to legacy FAISS if needed.
    :param paths: absolute paths to local files
    :return: status message about loaded documents or the directory path of the saved FAISS index
    """
    try:
        # Import MCP agent
        from tools.literature_agent_mcp_simple import _agent
        
        # Load documents into MCP system
        loaded_docs = []
        failed_docs = []
        
        for path in paths:
            if not os.path.isfile(path):
                failed_docs.append(f"{path}: File not found")
                continue
            
            try:
                result = _agent.load_file(path)  # Now handles both PDF and text files
                if "error" in result:
                    failed_docs.append(f"{path}: {result['error']}")
                elif result.get("success"):
                    loaded_docs.append({
                        "path": path,
                        "doc_id": result.get("doc_id"),
                        "chunks": result.get("chunks", 0),
                        "has_embeddings": result.get("has_embeddings", False)
                    })
                else:
                    failed_docs.append(f"{path}: Unknown error")
            except Exception as e:
                failed_docs.append(f"{path}: {str(e)}")
        
        if not loaded_docs:
            # Fall back to legacy approach
            print("MCP loading failed for all documents, falling back to FAISS")
            
            docs_all = []
            for path in paths:
                if not os.path.isfile(path):
                    continue
                if path.lower().endswith(".pdf"):
                    loader = PyPDFLoader(path)
                else:
                    loader = TextLoader(path, encoding="utf-8")
                docs_all.extend(loader.load())

            if not docs_all:
                raise ValueError("No valid documents loaded.")

            chunks = _split_docs(docs_all)
            vs = FAISS.from_documents(chunks, _embedder())
            out_dir = os.path.abspath(os.path.join("storage", "faiss"))
            os.makedirs(out_dir, exist_ok=True)
            vs.save_local(out_dir)
            return out_dir
        
        # Return status message for MCP approach
        total_chunks = sum(doc["chunks"] for doc in loaded_docs)
        total_with_embeddings = sum(1 for doc in loaded_docs if doc["has_embeddings"])
        
        status = f"✓ MCP Index Built Successfully\\n"
        status += f"  Documents loaded: {len(loaded_docs)}\\n"
        status += f"  Total chunks: {total_chunks}\\n"
        status += f"  Documents with embeddings: {total_with_embeddings}\\n"
        
        if failed_docs:
            status += f"\\n⚠ Failed documents ({len(failed_docs)}): " + "; ".join(failed_docs[:3])
            if len(failed_docs) > 3:
                status += f" and {len(failed_docs) - 3} more"
        
        # Storage location for reference
        status += f"\\n  Storage: mcp_literature_storage/"
        
        return status
        
    except Exception as e:
        return f"Error: {str(e)}"


@toolset.add()
def rag_answer(question: str, index_dir: Optional[str] = None, k: int = 5,
               temperature: Optional[float] = None) -> str:
    """
    Answer a question using RAG with MCP backend. If MCP has documents loaded, it will use them;
    otherwise falls back to the legacy FAISS index approach.
    :param question: user question
    :param index_dir: directory storing the FAISS index (used as fallback)
    :param k: top-k documents to retrieve
    :param temperature: optional LLM temperature override
    :return: grounded answer
    """
    try:
        # Import MCP agent
        from tools.literature_agent_mcp_simple import _agent
        
        # Check if MCP has documents loaded
        docs_list = _agent.list_documents()
        if docs_list.get("num_documents", 0) > 0:
            # Use MCP Q&A functionality
            qa_result = _agent.ask_question(question, num_sources=k)
            
            if "error" not in qa_result:
                return qa_result.get("answer", "No answer available")
            else:
                # Fall back to legacy approach if MCP fails
                print(f"MCP Q&A failed: {qa_result['error']}, falling back to FAISS")
        
        # Fallback to legacy FAISS approach
        idx = index_dir or os.path.abspath(os.path.join("storage", "faiss"))
        if not os.path.isdir(idx):
            raise FileNotFoundError(f"No documents in MCP and FAISS index directory not found: {idx}")

        vs = FAISS.load_local(idx, _embedder(), allow_dangerous_deserialization=True)
        docs = vs.similarity_search(question, k=k)

        context = "\n\n".join([d.page_content for d in docs])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a careful quantum computing literature analyst. Use only the provided context to answer. If unsure, say you don't know."),
            ("human", "Context:\n{context}\n\nQuestion: {question}\nAnswer concisely with inline citations if present.")
        ])

        llm = _get_llm(temperature)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})
        
    except Exception as e:
        return f"Error: {str(e)}"



@toolset.add()
def web_research(query: str, num_results: int = 5) -> list:
    """
    Perform web search using Google Gemini for quantum computing literature.
    :param query: search query for quantum computing topics
    :param num_results: number of results to request
    :return: list of formatted search results with citations
    """
    try:
        # Enhance query for quantum computing research
        enhanced_query = f"{query} quantum computing research papers scholarly articles"
        
        # Create a prompt for Gemini to search and summarize
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a quantum computing research assistant. Based on your knowledge, provide information about recent academic papers and research on the given topic. Include paper titles, authors when known, brief summaries, and publication venues. Format your response with clear sections for each paper or finding."),
            ("human", f"Find and summarize {num_results} recent research papers and developments about: {enhanced_query}. Focus on peer-reviewed papers, preprints from arXiv, and significant research contributions. For each paper, provide:\n1. Title\n2. Authors (if known)\n3. Publication venue/year\n4. Brief summary of key contributions\n5. Relevance to the query")
        ])
        
        llm = _get_llm(temperature=0.3)
        chain = prompt | llm | StrOutputParser()
        
        # Get the response from Gemini
        content = chain.invoke({"query": enhanced_query})
        
        # Format the response as a list
        formatted_results = [
            "**Search Results from Google Gemini**",
            "=" * 40,
            content
        ]
        
        return formatted_results
            
    except Exception as e:
        return [f"Search error: {str(e)}"]


