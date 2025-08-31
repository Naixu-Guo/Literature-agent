import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools.toolset import toolset


def _get_llm(temperature: Optional[float] = None) -> ChatOpenAI:
    temp = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.2"))
    model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    
    # Check if API key is valid
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-proj-"):
        # If using project key, try with gpt-3.5-turbo as fallback
        try:
            return ChatOpenAI(model=model, temperature=temp, api_key=api_key)
        except:
            # Fallback to cheaper model if quota exceeded
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=temp, api_key=api_key)
    
    return ChatOpenAI(model=model, temperature=temp, api_key=api_key)


def _embedder() -> OpenAIEmbeddings:
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=embed_model)


def _split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    return splitter.split_documents(docs)


@toolset.add()
def summarize_source(source: str, temperature: Optional[float] = None) -> str:
    """
    Summarize content from a local file or URL (PDF, text, or web page).
    :param source: absolute path to a local file (.pdf or .txt) or URL to a web page/PDF
    :param temperature: optional LLM temperature override
    :return: summary string
    """
    import tempfile
    import requests
    from langchain_community.document_loaders import WebBaseLoader
    from urllib.parse import urlparse
    
    # Check if source is a URL
    is_url = source.startswith(('http://', 'https://'))
    
    if is_url:
        # Handle PDF URLs
        if source.lower().endswith('.pdf') or 'pdf' in source.lower():
            try:
                # Download PDF to temporary file
                response = requests.get(source, timeout=30)
                response.raise_for_status()
                
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as tmp_file:
                    tmp_file.write(response.content)
                    tmp_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                # Clean up temp file
                os.unlink(tmp_path)
                
            except Exception as e:
                return f"Error downloading PDF from URL: {str(e)}"
        else:
            # Handle regular web pages
            try:
                loader = WebBaseLoader(source)
                docs = loader.load()
            except Exception as e:
                return f"Error loading web page: {str(e)}"
    else:
        # Handle local files
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")

        lower = source.lower()
        if lower.endswith(".pdf"):
            loader = PyPDFLoader(source)
        else:
            loader = TextLoader(source, encoding="utf-8")

        docs = loader.load()
    
    chunks = _split_docs(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a rigorous research assistant for quantum computing literature. "
                   "Write precise summaries with key findings, methods, datasets, and limitations. Be faithful and avoid hallucinations."),
        ("human", "Summarize the following content for a literature review. Provide bullet points with citations if present.\n{content}")
    ])

    try:
        llm = _get_llm(temperature)
        chain = prompt | llm | StrOutputParser()

        summaries = []
        for chunk in chunks:
            summaries.append(chain.invoke({"content": chunk.page_content}))
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            return f"OpenAI API quota exceeded. Please check your billing at https://platform.openai.com/account/billing or use a different API key."
        else:
            raise e

    joiner = "\n\n"
    return joiner.join(summaries)


@toolset.add()
def summarize_file(path: str, temperature: Optional[float] = None) -> str:
    """
    Legacy function: Summarize a local file (PDF or text).
    For URL support, use summarize_source instead.
    :param path: absolute path to a local file (.pdf or .txt)
    :param temperature: optional LLM temperature override
    :return: summary string
    """
    return summarize_source(path, temperature)


@toolset.add()
def summarize_multiple_sources(sources: list, temperature: Optional[float] = None) -> dict:
    """
    Summarize multiple sources (files and/or URLs) and return individual and combined summaries.
    :param sources: list of file paths or URLs
    :param temperature: optional LLM temperature override
    :return: dictionary with individual summaries and a combined summary
    """
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


@toolset.add()
def build_vector_index(paths: list) -> str:
    """
    Build a FAISS vector index from a list of local files (.pdf or .txt).
    Returns an index id saved to disk under ./storage/faiss.
    :param paths: absolute paths to local files
    :return: the directory path of the saved FAISS index
    """
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


@toolset.add()
def rag_answer(question: str, index_dir: Optional[str] = None, k: int = 5,
               temperature: Optional[float] = None) -> str:
    """
    Answer a question using RAG over a local FAISS index. If index_dir is not provided,
    the default ./storage/faiss is used.
    :param question: user question
    :param index_dir: directory storing the FAISS index
    :param k: top-k documents to retrieve
    :param temperature: optional LLM temperature override
    :return: grounded answer
    """
    idx = index_dir or os.path.abspath(os.path.join("storage", "faiss"))
    if not os.path.isdir(idx):
        raise FileNotFoundError(f"FAISS index directory not found: {idx}")

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


@toolset.add()
def web_research(query: str, num_results: int = 5) -> list:
    """
    Perform web search using Perplexity AI for quantum computing literature.
    :param query: search query for quantum computing topics
    :param num_results: number of results (not directly used with Perplexity)
    :return: list of formatted search results with citations
    """
    import requests
    
    perplexity_key = os.getenv("PERPLEXITY_API_KEY")
    if not perplexity_key or perplexity_key == "your-perplexity-api-key-here":
        return ["Please set PERPLEXITY_API_KEY in your .env file to use web search"]
    
    try:
        # Enhance query for quantum computing research
        enhanced_query = f"{query} quantum computing research papers scholarly articles"
        
        # Perplexity API endpoint
        url = "https://api.perplexity.ai/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {perplexity_key}",
            "Content-Type": "application/json"
        }
        
        # Configure the request for academic search
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a quantum computing research assistant. Search for and summarize recent academic papers and research. Include citations and links when available."
                },
                {
                    "role": "user",
                    "content": f"Find recent research papers and articles about: {enhanced_query}. Provide {num_results} relevant results with titles, authors (if available), brief summaries, and links."
                }
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 2000
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        # Check for errors
        if response.status_code != 200:
            error_detail = response.text
            return [f"Perplexity API error ({response.status_code}): {error_detail}"]
        
        response.raise_for_status()
        
        result = response.json()
        
        # Extract the response content
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            
            # Format the response as a list
            formatted_results = [
                "**Search Results from Perplexity AI**",
                "=" * 40,
                content
            ]
            
            return formatted_results
        else:
            return ["No results found from Perplexity API"]
            
    except requests.exceptions.RequestException as e:
        return [f"API request error: {str(e)}"]
    except Exception as e:
        return [f"Search error: {str(e)}"]


