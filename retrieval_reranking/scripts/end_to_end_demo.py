#!/usr/bin/env python3
"""
End-to-End Retrieval and Reranking Demo

This script demonstrates the complete pipeline:
1. Load pre-trained embeddings
2. Run queries through the system
3. Perform retrieval and reranking
4. Display top results with detailed metrics
5. Show component suggestions
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval.retrieval_pipeline import RetrievalPipeline
from utils.logging_utils import setup_logger, get_logger
from evaluation.benchmarking import RetrievalBenchmarker, create_test_queries

logger = get_logger(__name__)


def create_sample_documents() -> List[Dict[str, Any]]:
    """
    Create comprehensive sample documents for the demo.
    
    Returns:
        List of sample documents
    """
    return [
        {
            'id': 'webhook_trigger_guide',
            'text': 'Complete guide to creating webhook triggers in iFlow. Learn how to set up HTTP endpoints, configure authentication, handle incoming requests, and process webhook data. Includes examples for REST API integration and webhook validation.',
            'metadata': {
                'component_type': 'trigger',
                'complexity': 0.8,
                'reusability': 0.9,
                'freshness': 0.95,
                'popularity': 0.85,
                'tags': ['webhook', 'trigger', 'http', 'api', 'integration']
            }
        },
        {
            'id': 'trigger_setup',
            'text': 'Step-by-step tutorial for setting up various trigger types in iFlow. Covers timer triggers, file watchers, database triggers, and event-driven triggers. Includes configuration examples and best practices.',
            'metadata': {
                'component_type': 'trigger',
                'complexity': 0.6,
                'reusability': 0.8,
                'freshness': 0.9,
                'popularity': 0.75,
                'tags': ['trigger', 'setup', 'tutorial', 'configuration']
            }
        },
        {
            'id': 'api_integration',
            'text': 'Comprehensive guide to API integration in iFlow workflows. Learn how to make HTTP requests, handle responses, manage authentication, and implement error handling for external API calls.',
            'metadata': {
                'component_type': 'action',
                'complexity': 0.7,
                'reusability': 0.85,
                'freshness': 0.9,
                'popularity': 0.8,
                'tags': ['api', 'integration', 'http', 'requests', 'authentication']
            }
        },
        {
            'id': 'json_xml_converter',
            'text': 'Advanced data transformation component for converting between JSON and XML formats. Includes schema validation, custom mapping rules, and handling of complex nested structures. Supports both synchronous and asynchronous processing.',
            'metadata': {
                'component_type': 'transformer',
                'complexity': 0.9,
                'reusability': 0.95,
                'freshness': 0.85,
                'popularity': 0.7,
                'tags': ['json', 'xml', 'converter', 'transformation', 'schema']
            }
        },
        {
            'id': 'data_transformation',
            'text': 'General-purpose data transformation utilities for iFlow. Includes text processing, format conversion, data cleaning, and validation functions. Supports multiple input and output formats.',
            'metadata': {
                'component_type': 'transformer',
                'complexity': 0.7,
                'reusability': 0.9,
                'freshness': 0.8,
                'popularity': 0.75,
                'tags': ['transformation', 'data', 'processing', 'conversion']
            }
        },
        {
            'id': 'format_conversion',
            'text': 'Lightweight format conversion tools for common data formats. Supports CSV, JSON, XML, YAML, and custom delimited formats. Includes batch processing capabilities and error handling.',
            'metadata': {
                'component_type': 'transformer',
                'complexity': 0.5,
                'reusability': 0.8,
                'freshness': 0.9,
                'popularity': 0.65,
                'tags': ['format', 'conversion', 'csv', 'yaml', 'batch']
            }
        },
        {
            'id': 'error_handling',
            'text': 'Comprehensive error handling strategies for iFlow workflows. Covers exception management, retry mechanisms, fallback procedures, and error logging. Includes best practices for robust workflow design.',
            'metadata': {
                'component_type': 'error_handler',
                'complexity': 0.8,
                'reusability': 0.9,
                'freshness': 0.9,
                'popularity': 0.85,
                'tags': ['error', 'handling', 'exception', 'retry', 'logging']
            }
        },
        {
            'id': 'database_errors',
            'text': 'Specialized error handling for database operations in iFlow. Covers connection failures, query errors, transaction management, and data integrity issues. Includes recovery strategies and monitoring.',
            'metadata': {
                'component_type': 'error_handler',
                'complexity': 0.7,
                'reusability': 0.8,
                'freshness': 0.85,
                'popularity': 0.7,
                'tags': ['database', 'error', 'connection', 'transaction', 'recovery']
            }
        },
        {
            'id': 'connection_management',
            'text': 'Database connection management and pooling for iFlow workflows. Learn how to establish, maintain, and optimize database connections. Includes connection pooling, timeout handling, and resource management.',
            'metadata': {
                'component_type': 'connector',
                'complexity': 0.6,
                'reusability': 0.85,
                'freshness': 0.8,
                'popularity': 0.75,
                'tags': ['database', 'connection', 'pooling', 'management']
            }
        },
        {
            'id': 'field_mapping',
            'text': 'Advanced field mapping component for transforming data between different schemas. Supports complex mapping rules, conditional transformations, and data validation. Includes visual mapping interface and template management.',
            'metadata': {
                'component_type': 'data_mapper',
                'complexity': 0.8,
                'reusability': 0.95,
                'freshness': 0.9,
                'popularity': 0.8,
                'tags': ['field', 'mapping', 'schema', 'transformation', 'validation']
            }
        },
        {
            'id': 'schema_mapping',
            'text': 'Schema mapping utilities for converting between different data structures. Supports automatic schema detection, mapping generation, and validation. Includes support for complex nested objects and arrays.',
            'metadata': {
                'component_type': 'data_mapper',
                'complexity': 0.7,
                'reusability': 0.9,
                'freshness': 0.85,
                'popularity': 0.7,
                'tags': ['schema', 'mapping', 'detection', 'validation', 'nested']
            }
        },
        {
            'id': 'data_mapper',
            'text': 'Core data mapping functionality for iFlow workflows. Provides basic field mapping, data transformation, and format conversion capabilities. Suitable for simple to moderate complexity mapping requirements.',
            'metadata': {
                'component_type': 'data_mapper',
                'complexity': 0.5,
                'reusability': 0.8,
                'freshness': 0.9,
                'popularity': 0.75,
                'tags': ['data', 'mapper', 'field', 'transformation']
            }
        },
        {
            'id': 'data_aggregation',
            'text': 'Advanced data aggregation component for processing large datasets. Supports grouping, filtering, sorting, and statistical operations. Includes batch processing, memory optimization, and distributed processing capabilities.',
            'metadata': {
                'component_type': 'aggregator',
                'complexity': 0.9,
                'reusability': 0.9,
                'freshness': 0.85,
                'popularity': 0.7,
                'tags': ['aggregation', 'grouping', 'statistics', 'batch', 'distributed']
            }
        },
        {
            'id': 'batch_processing',
            'text': 'Batch processing utilities for handling large volumes of data in iFlow. Includes chunking, parallel processing, progress tracking, and error recovery. Supports various data sources and output formats.',
            'metadata': {
                'component_type': 'aggregator',
                'complexity': 0.7,
                'reusability': 0.85,
                'freshness': 0.9,
                'popularity': 0.75,
                'tags': ['batch', 'processing', 'chunking', 'parallel', 'progress']
            }
        },
        {
            'id': 'multi_source',
            'text': 'Multi-source data integration component for combining data from different sources. Supports data merging, conflict resolution, and unified output formatting. Includes data quality checks and validation.',
            'metadata': {
                'component_type': 'aggregator',
                'complexity': 0.8,
                'reusability': 0.9,
                'freshness': 0.85,
                'popularity': 0.7,
                'tags': ['multi-source', 'integration', 'merging', 'conflict', 'quality']
            }
        }
    ]


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_search_results(results: Dict[str, Any], query: str) -> None:
    """
    Print formatted search results.
    
    Args:
        results: Search results dictionary
        query: Original query
    """
    print_header(f"SEARCH RESULTS FOR: {query}")
    
    print(f"Total results: {results['total_results']}")
    print(f"Query: {results['query']}")
    print()
    
    if not results['results']:
        print("No results found.")
        return
    
    print("TOP RESULTS:")
    print("-" * 80)
    
    for i, result in enumerate(results['results'][:5], 1):
        doc = result.get('document', {}) if isinstance(result, dict) else {}
        doc_id = doc.get('id') or result.get('id', 'Unknown')
        # Try common text fields on the nested document, fall back to top-level
        text = (
            doc.get('text')
            or doc.get('content')
            or doc.get('body')
            or doc.get('description')
            or doc.get('title')
            or result.get('text', '')
        )
        similarity = result.get('similarity_score') or result.get('vector_score') or result.get('combined_score', 0.0)
        cross_enc = result.get('cross_encoder_score', 0.0)
        metadata_score = result.get('metadata_score', 0.0)
        final_score = result.get('final_score') or result.get('hybrid_score') or result.get('score', 0.0)

        print(f"{i}. {doc_id}")
        print(f"   Text: {str(text)[:100]}...")
        print(f"   Similarity Score: {float(similarity):.3f}")
        print(f"   Cross-Encoder Score: {float(cross_enc):.3f}")
        print(f"   Metadata Score: {float(metadata_score):.3f}")
        print(f"   Final Score: {float(final_score):.3f}")
        
        metadata = doc.get('metadata', {}) if isinstance(doc, dict) else {}
        if metadata:
            print(f"   Component Type: {metadata.get('component_type', 'Unknown')}")
            print(f"   Complexity: {metadata.get('complexity', 0.0):.2f}")
            print(f"   Reusability: {metadata.get('reusability', 0.0):.2f}")
        
        print()


def print_component_suggestions(suggestions: List[Dict[str, Any]]) -> None:
    """
    Print component suggestions.
    
    Args:
        suggestions: List of component suggestions
    """
    if not suggestions:
        print("No component suggestions available.")
        return
    
    print_header("COMPONENT SUGGESTIONS")
    
    for i, suggestion in enumerate(suggestions[:3], 1):
        print(f"{i}. {suggestion['component_type'].upper()}")
        print(f"   Confidence: {suggestion['confidence']:.3f}")
        print(f"   Description: {suggestion['description']}")
        print(f"   Examples: {', '.join(suggestion['examples'][:3])}")
        print()


def print_pipeline_stats(stats: Dict[str, Any]) -> None:
    """
    Print pipeline statistics.
    
    Args:
        stats: Pipeline statistics
    """
    print_header("PIPELINE STATISTICS")
    
    # Vector search stats
    vector_stats = stats.get('vector_search', {})
    print("Vector Search:")
    print(f"  Total documents: {vector_stats.get('total_documents', 0)}")
    print(f"  Index size: {vector_stats.get('index_size', 0)}")
    print(f"  Embedding dimension: {vector_stats.get('embedding_dimension', 0)}")
    print(f"  Model: {vector_stats.get('model_name', 'Unknown')}")
    print()
    
    # Pipeline configuration
    pipeline_config = stats.get('pipeline_config', {})
    print("Pipeline Configuration:")
    for key, value in pipeline_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Component status
    components = stats.get('components', {})
    print("Component Status:")
    for component, enabled in components.items():
        status = "✓ Enabled" if enabled else "✗ Disabled"
        print(f"  {component}: {status}")


def run_interactive_demo() -> None:
    """Run interactive demo with user queries."""
    print_header("INTERACTIVE DEMO")
    print("Enter queries to test the system. Type 'quit' to exit.")
    print()
    
    while True:
        try:
            query = input("Enter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print(f"\nProcessing query: {query}")
            print("-" * 40)
            
            # Time the search
            start_time = time.time()
            results = pipeline.search(query, top_k=5, return_breakdown=True)
            search_time = time.time() - start_time
            
            print(f"Search completed in {search_time:.3f} seconds")
            print()
            
            # Print results
            print_search_results(results, query)
            
            # Print component suggestions
            suggestions = pipeline.get_component_suggestions(query)
            print_component_suggestions(suggestions)
            
            # Print score breakdown if available
            if 'score_breakdown' in results:
                breakdown = results['score_breakdown']
                print_header("SCORE BREAKDOWN (Top Result)")
                print(f"Vector Score:      {breakdown['vector_score']:.3f}")
                print(f"Cross-Encoder:     {breakdown['cross_encoder_score']:.3f}")
                print(f"Metadata Score:    {breakdown['metadata_score']:.3f}")
                print(f"Final Score:       {breakdown['final_score']:.3f}")
                print()
            
        except KeyboardInterrupt:
            print("\nDemo interrupted by user.")
            break
        except Exception as e:
            print(f"Error processing query: {e}")
            logger.error(f"Error in interactive demo: {e}")


def run_benchmark_demo() -> None:
    """Run benchmark demo with predefined test queries."""
    print_header("BENCHMARK DEMO")
    
    # Create benchmarker
    benchmarker = RetrievalBenchmarker()
    
    # Create test queries
    test_queries = create_test_queries()
    
    print(f"Running benchmark with {len(test_queries)} test queries...")
    print()
    
    # Run benchmark
    results = benchmarker.benchmark_retrieval_system(pipeline, test_queries)
    
    # Generate and print performance report
    report = benchmarker.generate_performance_report(results)
    print(report)
    
    # Save results
    benchmarker.save_results(results, "performance_report.json")
    
    print(f"\nDetailed results saved to: {benchmarker.results_dir}/performance_report.json")


def main():
    """Main demo function."""
    global pipeline
    
    print_header("END-TO-END RETRIEVAL AND RERANKING DEMO")
    print("This demo showcases the complete pipeline with OpenAI-powered components.")
    print()
    
    # Setup logging
    setup_logger("demo", log_file="demo.log")
    
    try:
        # Initialize pipeline
        print("Initializing retrieval pipeline...")
        pipeline = RetrievalPipeline(
            enable_cross_encoder=True,
            enable_metadata_reranking=True,
            enable_query_classification=True,
            enable_hybrid_reranking=True
        )
        
        # Create and add sample documents
        print("Creating sample documents...")
        documents = create_sample_documents()
        pipeline.add_documents(documents)
        
        print(f"Added {len(documents)} documents to the pipeline.")
        print()
        
        # Print pipeline statistics
        stats = pipeline.get_pipeline_stats()
        print_pipeline_stats(stats)
        
        # Demo options
        while True:
            print_header("DEMO OPTIONS")
            print("1. Interactive Query Demo")
            print("2. Benchmark Demo")
            print("3. Exit")
            print()
            
            choice = input("Select an option (1-3): ").strip()
            
            if choice == '1':
                run_interactive_demo()
            elif choice == '2':
                run_benchmark_demo()
            elif choice == '3':
                print("Exiting demo...")
                break
            else:
                print("Invalid option. Please select 1, 2, or 3.")
                print()
        
    except Exception as e:
        print(f"Error in demo: {e}")
        logger.error(f"Demo error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
