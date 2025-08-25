#!/usr/bin/env python3
"""
Demo script for the retrieval and reranking system.
This script demonstrates how to use the complete pipeline with example data.
"""

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval.retrieval_pipeline import RetrievalPipeline
from utils.logging_utils import setup_logger


def create_sample_documents():
    """Create sample documents for demonstration."""
    return [
        {
            "id": "1",
            "title": "HTTP Trigger Configuration",
            "text": "Configure HTTP triggers to handle incoming web requests. Set up endpoints, authentication, and response handling.",
            "metadata": {
                "component_type": "trigger",
                "complexity": "medium",
                "created_date": "2024-01-15",
                "views": 1500,
                "rating": 4.5
            }
        },
        {
            "id": "2",
            "title": "Data Transformation Pipeline",
            "text": "Transform data between different formats using JSON transformers and data mappers. Handle field mapping and validation.",
            "metadata": {
                "component_type": "transformer",
                "complexity": "high",
                "created_date": "2024-01-20",
                "views": 2200,
                "rating": 4.8
            }
        },
        {
            "id": "3",
            "title": "Error Handling Best Practices",
            "text": "Implement robust error handling with retry mechanisms, logging, and fallback strategies for production systems.",
            "metadata": {
                "component_type": "error_handler",
                "complexity": "medium",
                "created_date": "2024-01-10",
                "views": 1800,
                "rating": 4.6
            }
        },
        {
            "id": "4",
            "title": "Database Connector Setup",
            "text": "Connect to various databases using connectors. Configure authentication, connection pooling, and query optimization.",
            "metadata": {
                "component_type": "connector",
                "complexity": "high",
                "created_date": "2024-01-25",
                "views": 3000,
                "rating": 4.9
            }
        },
        {
            "id": "5",
            "title": "Conditional Logic Implementation",
            "text": "Use if conditions and switch routers to implement business logic and route data based on specific criteria.",
            "metadata": {
                "component_type": "condition",
                "complexity": "low",
                "created_date": "2024-01-05",
                "views": 1200,
                "rating": 4.3
            }
        },
        {
            "id": "6",
            "title": "Batch Data Processing",
            "text": "Process large datasets in batches using aggregators and collectors. Optimize performance and memory usage.",
            "metadata": {
                "component_type": "aggregator",
                "complexity": "high",
                "created_date": "2024-01-30",
                "views": 2500,
                "rating": 4.7
            }
        },
        {
            "id": "7",
            "title": "API Integration Guide",
            "text": "Integrate with external APIs using HTTP requests and authentication. Handle rate limiting and error responses.",
            "metadata": {
                "component_type": "action",
                "complexity": "medium",
                "created_date": "2024-01-12",
                "views": 1900,
                "rating": 4.4
            }
        },
        {
            "id": "8",
            "title": "Data Mapping Tutorial",
            "text": "Learn how to map fields between different data schemas. Use data mappers for ETL processes and integrations.",
            "metadata": {
                "component_type": "data_mapper",
                "complexity": "medium",
                "created_date": "2024-01-18",
                "views": 1600,
                "rating": 4.2
            }
        }
    ]


def run_demo():
    """Run the complete retrieval and reranking demo."""
    print("🚀 Starting Retrieval and Reranking System Demo")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logger("demo", log_file="demo.log")
    
    # Initialize the pipeline
    print("\n1. Initializing Retrieval Pipeline...")
    pipeline = RetrievalPipeline(
        index_path="data/demo_indices",
        enable_cross_encoder=True,
        enable_metadata_reranking=True,
        enable_query_classification=True,
        enable_hybrid_reranking=True,
        log_file="demo.log"
    )
    
    # Add sample documents
    print("\n2. Adding Sample Documents...")
    documents = create_sample_documents()
    pipeline.add_documents(documents)
    print(f"✅ Added {len(documents)} documents to the index")
    
    # Example queries
    test_queries = [
        "How to handle HTTP requests and webhooks?",
        "Transform JSON data between different formats",
        "Implement error handling with retry logic",
        "Connect to database and perform queries",
        "Use conditional logic for business rules",
        "Process large datasets in batches",
        "Integrate with external APIs",
        "Map fields between different schemas"
    ]
    
    print("\n3. Running Search Examples...")
    print("-" * 40)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 30)
        
        # Perform search
        results = pipeline.search(
            query=query,
            top_k=3,
            threshold=0.1,
            enable_reranking=True,
            return_breakdown=True
        )
        
        # Display results
        print(f"Found {results['total_results']} results")
        
        if 'query_classification' in results and results['query_classification']:
            classification = results['query_classification']
            print(f"Query Type: {classification['primary_type']} (confidence: {classification.get('confidence', 0.0):.2f})")
        
        for j, result in enumerate(results['results'][:3], 1):
            doc = result['document']
            print(f"  {j}. {doc['title']} (Score: {result['score']:.3f})")
            print(f"     Component: {doc['metadata']['component_type']}")
            print(f"     Complexity: {doc['metadata']['complexity']}")
        
        # Show component suggestions
        suggestions = pipeline.get_component_suggestions(query)
        if suggestions:
            print(f"  Suggested Components:")
            for suggestion in suggestions[:2]:
                component_type = suggestion.get('component_type', suggestion.get('component', 'Unknown'))
                description = suggestion.get('description', 'No description available')
                print(f"    - {component_type}: {description}")
    
    # Show pipeline statistics
    print("\n4. Pipeline Statistics...")
    print("-" * 40)
    stats = pipeline.get_pipeline_stats()
    
    print(f"Vector Search: {stats['vector_search']['total_documents']} documents")
    print(f"Model: {stats['vector_search']['model_name']}")
    print(f"Cross-encoder: {'Enabled' if stats['pipeline_config']['enable_cross_encoder'] else 'Disabled'}")
    print(f"Metadata Reranking: {'Enabled' if stats['pipeline_config']['enable_metadata_reranking'] else 'Disabled'}")
    print(f"Query Classification: {'Enabled' if stats['pipeline_config']['enable_query_classification'] else 'Disabled'}")
    print(f"Hybrid Reranking: {'Enabled' if stats['pipeline_config']['enable_hybrid_reranking'] else 'Disabled'}")
    
    # Save pipeline
    print("\n5. Saving Pipeline...")
    pipeline.save_pipeline()
    print("✅ Pipeline saved successfully")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("- Check the 'data/demo_indices' directory for saved models")
    print("- Review 'demo.log' for detailed execution logs")
    print("- Try different queries and configurations")
    print("- Experiment with enabling/disabling components")


def run_comparison_demo():
    """Run a comparison demo showing different ranking approaches."""
    print("\n🔍 Running Ranking Comparison Demo")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = RetrievalPipeline(
        index_path="data/comparison_demo",
        enable_cross_encoder=True,
        enable_metadata_reranking=True,
        enable_query_classification=True,
        enable_hybrid_reranking=True
    )
    
    # Add documents
    documents = create_sample_documents()
    pipeline.add_documents(documents)
    
    # Test query
    query = "How to handle errors and implement retry logic?"
    
    print(f"\nQuery: {query}")
    print("-" * 40)
    
    # Test different configurations
    configurations = [
        ("Vector Search Only", {"enable_reranking": False}),
        ("With Cross-Encoder", {"enable_reranking": True}),
        ("With Metadata Reranking", {"enable_reranking": True}),
        ("Full Hybrid", {"enable_reranking": True})
    ]
    
    for config_name, config_params in configurations:
        print(f"\n{config_name}:")
        print("-" * 20)
        
        results = pipeline.search(
            query=query,
            top_k=3,
            **config_params
        )
        
        for i, result in enumerate(results['results'][:3], 1):
            doc = result['document']
            print(f"  {i}. {doc['title']} (Score: {result['score']:.3f})")


if __name__ == "__main__":
    try:
        # Run main demo
        run_demo()
        
        # Run comparison demo
        run_comparison_demo()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
