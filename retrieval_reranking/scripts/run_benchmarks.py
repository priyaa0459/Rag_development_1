#!/usr/bin/env python3
"""
Comprehensive Benchmarking Script

This script runs all benchmarks for the retrieval and reranking system:
1. Precision@k, Recall@k, MRR metrics
2. Latency benchmarks (retrieval & reranking times)
3. Reranker comparisons
4. Performance reports
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


def create_benchmark_documents() -> List[Dict[str, Any]]:
    """
    Create comprehensive benchmark documents.
    
    Returns:
        List of benchmark documents
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
                'popularity': 0.85
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
                'popularity': 0.75
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
                'popularity': 0.8
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
                'popularity': 0.7
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
                'popularity': 0.75
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
                'popularity': 0.65
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
                'popularity': 0.85
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
                'popularity': 0.7
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
                'popularity': 0.75
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
                'popularity': 0.8
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
                'popularity': 0.7
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
                'popularity': 0.75
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
                'popularity': 0.7
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
                'popularity': 0.75
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
                'popularity': 0.7
            }
        }
    ]


def create_reranker_configs() -> List[Dict[str, Any]]:
    """
    Create different reranker configurations for comparison.
    
    Returns:
        List of reranker configurations
    """
    return [
        {
            'name': 'Vector_Only',
            'config': {
                'enable_cross_encoder': False,
                'enable_metadata_reranking': False,
                'enable_query_classification': False,
                'enable_hybrid_reranking': False
            }
        },
        {
            'name': 'Vector_CrossEncoder',
            'config': {
                'enable_cross_encoder': True,
                'enable_metadata_reranking': False,
                'enable_query_classification': False,
                'enable_hybrid_reranking': False
            }
        },
        {
            'name': 'Vector_Metadata',
            'config': {
                'enable_cross_encoder': False,
                'enable_metadata_reranking': True,
                'enable_query_classification': False,
                'enable_hybrid_reranking': False
            }
        },
        {
            'name': 'Vector_Classification',
            'config': {
                'enable_cross_encoder': False,
                'enable_metadata_reranking': False,
                'enable_query_classification': True,
                'enable_hybrid_reranking': False
            }
        },
        {
            'name': 'Full_Hybrid',
            'config': {
                'enable_cross_encoder': True,
                'enable_metadata_reranking': True,
                'enable_query_classification': True,
                'enable_hybrid_reranking': True
            }
        }
    ]


def run_latency_benchmarks(pipeline, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run detailed latency benchmarks.
    
    Args:
        pipeline: Retrieval pipeline
        test_queries: List of test queries
        
    Returns:
        Latency benchmark results
    """
    logger.info("Running latency benchmarks...")
    
    latency_results = {
        'retrieval_times': [],
        'reranking_times': [],
        'total_times': [],
        'component_times': {
            'vector_search': [],
            'cross_encoder': [],
            'metadata_reranker': [],
            'query_classifier': [],
            'hybrid_reranker': []
        }
    }
    
    for i, test_query in enumerate(test_queries):
        query = test_query['query']
        logger.info(f"Latency benchmark {i+1}/{len(test_queries)}: {query[:50]}...")
        
        # Measure retrieval time
        start_time = time.time()
        retrieval_results = pipeline.search(query, top_k=10, enable_reranking=False)
        retrieval_time = time.time() - start_time
        
        # Measure reranking time
        start_time = time.time()
        reranking_results = pipeline.search(query, top_k=10, enable_reranking=True)
        reranking_time = time.time() - start_time
        
        total_time = retrieval_time + reranking_time
        
        latency_results['retrieval_times'].append(retrieval_time)
        latency_results['reranking_times'].append(reranking_time)
        latency_results['total_times'].append(total_time)
    
    # Calculate statistics
    for time_type in ['retrieval_times', 'reranking_times', 'total_times']:
        times = latency_results[time_type]
        latency_results[f'{time_type}_avg'] = sum(times) / len(times)
        latency_results[f'{time_type}_min'] = min(times)
        latency_results[f'{time_type}_max'] = max(times)
        latency_results[f'{time_type}_std'] = (sum((t - latency_results[f'{time_type}_avg']) ** 2 for t in times) / len(times)) ** 0.5
    
    return latency_results


def run_comprehensive_benchmarks() -> None:
    """Run all comprehensive benchmarks."""
    print("=" * 80)
    print("COMPREHENSIVE RETRIEVAL AND RERANKING BENCHMARKS")
    print("=" * 80)
    print()
    
    # Setup logging
    setup_logger("benchmarks", log_file="benchmarks.log")
    
    try:
        # Initialize pipeline
        print("Initializing retrieval pipeline...")
        pipeline = RetrievalPipeline(
            enable_cross_encoder=True,
            enable_metadata_reranking=True,
            enable_query_classification=True,
            enable_hybrid_reranking=True
        )
        
        # Add benchmark documents
        print("Adding benchmark documents...")
        documents = create_benchmark_documents()
        pipeline.add_documents(documents)
        print(f"Added {len(documents)} documents to the pipeline.")
        print()
        
        # Create benchmarker
        benchmarker = RetrievalBenchmarker()
        
        # Create test queries
        test_queries = create_test_queries()
        print(f"Created {len(test_queries)} test queries for benchmarking.")
        print()
        
        # 1. Run comprehensive system benchmark
        print("1. Running comprehensive system benchmark...")
        system_results = benchmarker.benchmark_retrieval_system(pipeline, test_queries)
        
        # Generate and print performance report
        report = benchmarker.generate_performance_report(system_results)
        print(report)
        
        # Save system results
        benchmarker.save_results(system_results, "comprehensive_benchmark.json")
        print(f"System benchmark results saved to: {benchmarker.results_dir}/comprehensive_benchmark.json")
        print()
        
        # 2. Run latency benchmarks
        print("2. Running latency benchmarks...")
        latency_results = run_latency_benchmarks(pipeline, test_queries)
        
        # Print latency report
        print("LATENCY BENCHMARK RESULTS:")
        print("-" * 40)
        print(f"Retrieval Time:  {latency_results['retrieval_times_avg']:.3f}s ± {latency_results['retrieval_times_std']:.3f}s")
        print(f"Reranking Time:  {latency_results['reranking_times_avg']:.3f}s ± {latency_results['reranking_times_std']:.3f}s")
        print(f"Total Time:      {latency_results['total_times_avg']:.3f}s ± {latency_results['total_times_std']:.3f}s")
        print(f"Throughput:      {len(test_queries) / sum(latency_results['total_times']):.2f} queries/second")
        print()
        
        # Save latency results
        benchmarker.save_results(latency_results, "latency_benchmark.json")
        print(f"Latency benchmark results saved to: {benchmarker.results_dir}/latency_benchmark.json")
        print()
        
        # 3. Run reranker comparison
        print("3. Running reranker comparison...")
        reranker_configs = create_reranker_configs()
        comparison_results = benchmarker.compare_rerankers(pipeline, test_queries, reranker_configs)
        
        # Print comparison results
        print("RERANKER COMPARISON RESULTS:")
        print("-" * 40)
        for config_name, metrics in comparison_results['overall_metrics'].items():
            print(f"{config_name}:")
            print(f"  Precision@5: {metrics['precision_at_5_avg']:.3f}")
            print(f"  Recall@5:    {metrics['recall_at_5_avg']:.3f}")
            print(f"  MRR:         {metrics['mrr_avg']:.3f}")
            print(f"  Avg Time:    {metrics['avg_total_time']:.3f}s")
            print()
        
        # Save comparison results
        benchmarker.save_results(comparison_results, "reranker_comparison.json")
        print(f"Reranker comparison results saved to: {benchmarker.results_dir}/reranker_comparison.json")
        print()
        
        # 4. Generate final performance report
        print("4. Generating final performance report...")
        final_report = {
            'system_benchmark': system_results,
            'latency_benchmark': latency_results,
            'reranker_comparison': comparison_results,
            'summary': {
                'total_documents': len(documents),
                'total_queries': len(test_queries),
                'best_reranker': max(comparison_results['overall_metrics'].items(), 
                                   key=lambda x: x[1]['mrr_avg'])[0],
                'fastest_reranker': min(comparison_results['overall_metrics'].items(), 
                                      key=lambda x: x[1]['avg_total_time'])[0]
            }
        }
        
        benchmarker.save_results(final_report, "performance_report.json")
        print(f"Final performance report saved to: {benchmarker.results_dir}/performance_report.json")
        print()
        
        # Print summary
        print("BENCHMARK SUMMARY:")
        print("-" * 40)
        print(f"Best performing reranker: {final_report['summary']['best_reranker']}")
        print(f"Fastest reranker: {final_report['summary']['fastest_reranker']}")
        print(f"Total documents indexed: {final_report['summary']['total_documents']}")
        print(f"Total queries tested: {final_report['summary']['total_queries']}")
        print()
        
        print("All benchmarks completed successfully!")
        print(f"Results saved in: {benchmarker.results_dir}")
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        logger.error(f"Benchmark error: {e}")
        return 1
    
    return 0


def main():
    """Main function."""
    return run_comprehensive_benchmarks()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
