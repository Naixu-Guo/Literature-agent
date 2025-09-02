#!/usr/bin/env python3
"""
Comprehensive test suite for Literature Agent system
Tests all major functionality to ensure maximum output quality
"""

import json
import time
from tools.literature_agent import (
    summarize_source,
    query_documents,
    web_research,
    list_documents,
    check_embeddings_health,
    create_embeddings
)

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_document_loading():
    """Test document loading from various sources"""
    print_section("TEST 1: Document Loading")
    
    # Test loading from arXiv
    print("\n1. Loading quantum algorithm papers from arXiv...")
    papers = [
        "https://arxiv.org/abs/0811.3171",  # HHL Algorithm
        "https://arxiv.org/abs/quant-ph/9511026",  # QPE (Kitaev)
        "https://arxiv.org/abs/1511.04206"  # QSVT
    ]
    
    for paper in papers:
        print(f"\n  Loading: {paper}")
        result = web_research(paper)
        if "Successfully loaded" in result:
            print("  ✅ Loaded successfully")
        else:
            print(f"  ❌ Failed: {result[:100]}")
    
    # Check documents loaded
    docs = list_documents()
    doc_count = docs.count("ID:")
    print(f"\n  Total documents in storage: {doc_count}")
    return doc_count > 0

def test_embeddings():
    """Test embedding creation and health"""
    print_section("TEST 2: Embeddings System")
    
    # Check embeddings health
    print("\n1. Checking embeddings health...")
    health = check_embeddings_health()
    print(f"  {health[:200]}")
    
    # Create embeddings if needed
    if "incomplete" in health or "error" in health:
        print("\n2. Creating embeddings for all documents...")
        result = create_embeddings("all")
        print(f"  {result}")
        
        # Recheck health
        health = check_embeddings_health()
        print(f"\n3. Rechecking health: {health[:200]}")
    
    return "✅" in health

def test_algorithm_extraction():
    """Test comprehensive algorithm extraction"""
    print_section("TEST 3: Algorithm Extraction Quality")
    
    test_queries = [
        {
            "query": "HHL algorithm for solving linear systems of equations",
            "expected_fields": ["algorithm_name", "problem_description", "resources", "subroutines"]
        },
        {
            "query": "Quantum Phase Estimation algorithm implementation",
            "expected_fields": ["algorithm_name", "computational_model", "algorithm_steps"]
        },
        {
            "query": "Grover's search algorithm complexity and resource requirements",
            "expected_fields": ["algorithm_name", "resources", "success_probability"]
        }
    ]
    
    all_passed = True
    for i, test in enumerate(test_queries, 1):
        print(f"\n{i}. Testing: {test['query'][:50]}...")
        
        result = query_documents(test['query'], mode='algorithm_spec', num_results=10)
        
        if isinstance(result, dict) and 'algorithm_spec' in result:
            spec = result['algorithm_spec']
            
            # Check if key fields are populated
            populated_fields = []
            empty_fields = []
            
            for field in test['expected_fields']:
                if field in spec:
                    value = spec[field]
                    if value and value != "Information not available in the provided context.":
                        populated_fields.append(field)
                    else:
                        empty_fields.append(field)
            
            if populated_fields:
                print(f"  ✅ Populated fields: {', '.join(populated_fields)}")
            if empty_fields:
                print(f"  ⚠️  Empty fields: {', '.join(empty_fields)}")
            
            # Print sample of extracted data
            if 'algorithm_name' in spec and spec['algorithm_name']:
                print(f"  Algorithm: {spec['algorithm_name']}")
            if 'resources' in spec and isinstance(spec['resources'], dict):
                resources = spec['resources']
                if resources.get('logical_qubits'):
                    print(f"  Logical qubits: {resources['logical_qubits']}")
                if resources.get('runtime_complexity'):
                    print(f"  Runtime: {resources['runtime_complexity']}")
        else:
            print(f"  ❌ Failed to extract algorithm spec")
            all_passed = False
    
    return all_passed

def test_answer_mode():
    """Test answer mode for general queries"""
    print_section("TEST 4: Answer Mode Quality")
    
    queries = [
        "What is the computational complexity of the HHL algorithm?",
        "How many qubits are required for Quantum Phase Estimation?",
        "What are the main applications of quantum algorithms for linear systems?"
    ]
    
    all_passed = True
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: {query[:50]}...")
        
        result = query_documents(query, mode='answer', num_results=8)
        
        if isinstance(result, str) and len(result) > 50:
            print(f"  ✅ Answer generated ({len(result)} chars)")
            print(f"  Preview: {result[:150]}...")
        else:
            print(f"  ❌ Failed to generate answer")
            all_passed = False
    
    return all_passed

def test_comprehensive_extraction():
    """Test extraction with maximum detail for resource estimation"""
    print_section("TEST 5: Comprehensive Resource Estimation Data")
    
    print("\n1. Extracting detailed HHL algorithm specification...")
    
    result = query_documents(
        "HHL algorithm complete specification with all resource requirements, "
        "gate counts, circuit depth, and implementation details for resource estimation",
        mode='algorithm_spec',
        num_results=15  # Use more sources for comprehensive extraction
    )
    
    if isinstance(result, dict) and 'algorithm_spec' in result:
        spec = result['algorithm_spec']
        
        # Count populated vs empty fields
        def count_fields(obj, prefix=""):
            populated = 0
            total = 0
            details = []
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    total += 1
                    full_key = f"{prefix}.{key}" if prefix else key
                    
                    if isinstance(value, dict):
                        sub_pop, sub_total, sub_details = count_fields(value, full_key)
                        populated += sub_pop
                        total += sub_total - 1  # Don't double count the dict itself
                        details.extend(sub_details)
                    elif isinstance(value, list) and value:
                        populated += 1
                        details.append(f"{full_key}: {len(value)} items")
                    elif value and value not in [None, "", "Information not available in the provided context."]:
                        populated += 1
                        if isinstance(value, str):
                            details.append(f"{full_key}: {value[:50]}...")
                        else:
                            details.append(f"{full_key}: {value}")
            
            return populated, total, details
        
        populated, total, details = count_fields(spec)
        
        print(f"\n  📊 Extraction Statistics:")
        print(f"  - Fields populated: {populated}/{total} ({100*populated//total}%)")
        print(f"  - Sources used: {len(result.get('sources', []))}")
        
        print(f"\n  📋 Sample Extracted Data:")
        for detail in details[:15]:  # Show first 15 populated fields
            print(f"    • {detail}")
        
        # Generate resource estimation summary
        print(f"\n  🎯 Resource Estimation Summary:")
        resources = spec.get('resources', {})
        if resources:
            for key, value in resources.items():
                if value and value not in [None, "Information not available in the provided context."]:
                    print(f"    • {key}: {value}")
        
        return populated > total * 0.3  # Success if >30% fields populated
    else:
        print("  ❌ Failed to extract comprehensive specification")
        return False

def main():
    """Run all tests and report results"""
    print("\n" + "="*60)
    print("  LITERATURE AGENT COMPREHENSIVE TEST SUITE")
    print("  Testing Maximum Output Quality for Query Documents")
    print("="*60)
    
    # Track test results
    results = {}
    
    # Run tests
    print("\n🚀 Starting comprehensive system test...")
    
    results['loading'] = test_document_loading()
    time.sleep(1)  # Brief pause between tests
    
    results['embeddings'] = test_embeddings()
    time.sleep(1)
    
    results['extraction'] = test_algorithm_extraction()
    time.sleep(1)
    
    results['answers'] = test_answer_mode()
    time.sleep(1)
    
    results['comprehensive'] = test_comprehensive_extraction()
    
    # Final report
    print_section("FINAL TEST REPORT")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    print(f"\n📊 Test Results Summary:")
    print(f"  • Document Loading: {'✅ PASSED' if results['loading'] else '❌ FAILED'}")
    print(f"  • Embeddings System: {'✅ PASSED' if results['embeddings'] else '❌ FAILED'}")
    print(f"  • Algorithm Extraction: {'✅ PASSED' if results['extraction'] else '❌ FAILED'}")
    print(f"  • Answer Mode: {'✅ PASSED' if results['answers'] else '❌ FAILED'}")
    print(f"  • Comprehensive Extraction: {'✅ PASSED' if results['comprehensive'] else '❌ FAILED'}")
    
    print(f"\n🎯 Overall Score: {passed_tests}/{total_tests} tests passed ({100*passed_tests//total_tests}%)")
    
    if passed_tests == total_tests:
        print("\n✨ SUCCESS! All systems functioning optimally for maximum output quality!")
        print("   The algorithm_spec mode is ready for comprehensive quantum algorithm extraction")
        print("   suitable for resource estimation tasks.")
    else:
        print("\n⚠️  Some tests failed. Review the output above for details.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()