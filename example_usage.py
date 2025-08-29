#!/usr/bin/env python3
"""
Example usage of the Literature Review Agent for Quantum Computing
"""

import os
import sys
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tools.literature_agent import (
    summarize_file,
    build_vector_index,
    rag_answer,
    web_research
)


def main():
    """Demonstrate the literature agent capabilities"""
    
    print("=" * 60)
    print("Literature Review Agent - Example Usage")
    print("=" * 60)
    
    # Example 1: Web Research
    print("\n1. Searching for Quantum Computing Literature Online...")
    print("-" * 40)
    
    try:
        search_results = web_research("quantum error correction 2024", num_results=3)
        for i, result in enumerate(search_results, 1):
            print(f"\nResult {i}:")
            print(result)
            print("-" * 20)
    except Exception as e:
        print(f"Web search error: {e}")
    
    # Example 2: Document Processing (if PDFs are available)
    print("\n2. Document Processing Example")
    print("-" * 40)
    
    # Check if there are any PDFs in the current directory
    pdf_files = list(Path(".").glob("*.pdf"))
    
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF files")
        
        # Summarize first PDF
        first_pdf = str(pdf_files[0].absolute())
        print(f"\nSummarizing: {pdf_files[0].name}")
        
        try:
            summary = summarize_file(first_pdf, temperature=0.3)
            print("\nSummary:")
            print(summary[:500] + "..." if len(summary) > 500 else summary)
        except Exception as e:
            print(f"Summarization error: {e}")
        
        # Build vector index if multiple PDFs
        if len(pdf_files) > 1:
            print("\n3. Building Vector Index from Multiple Documents")
            print("-" * 40)
            
            try:
                pdf_paths = [str(p.absolute()) for p in pdf_files[:3]]  # Limit to 3 files
                index_dir = build_vector_index(pdf_paths)
                print(f"Vector index created at: {index_dir}")
                
                # Example RAG query
                print("\n4. Asking Questions using RAG")
                print("-" * 40)
                
                question = "What are the main challenges in quantum computing?"
                answer = rag_answer(question, index_dir=index_dir, k=3)
                print(f"Question: {question}")
                print(f"Answer: {answer}")
                
            except Exception as e:
                print(f"Vector index/RAG error: {e}")
    else:
        print("No PDF files found in current directory.")
        print("To test document processing features, place PDF files in the current directory.")
    
    # Example 3: Research workflow
    print("\n5. Complete Research Workflow Example")
    print("-" * 40)
    print("""
    Typical workflow:
    1. Search for papers online using web_research()
    2. Download relevant PDFs
    3. Build vector index with build_vector_index()
    4. Summarize key papers with summarize_file()
    5. Ask specific questions with rag_answer()
    """)
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set!")
        print("Please set it before running: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    
    main()