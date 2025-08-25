import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os

from utils.logging_utils import get_logger, RetrievalLogger
from utils.scoring_utils import calculate_openai_similarity

logger = get_logger(__name__)


class RetrievalBenchmarker:
    """
    Comprehensive benchmarking for retrieval and reranking systems.
    """
    
    def __init__(self, results_dir: str = "./results"):
        """
        Initialize the benchmarker.
        
        Args:
            results_dir: Directory to save benchmark results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Benchmarker initialized with results directory: {results_dir}")
    
    def calculate_precision_at_k(self, relevant_docs: List[str], retrieved_docs: List[str], k: int) -> float:
        """
        Calculate precision at k.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            k: Number of top results to consider
            
        Returns:
            Precision at k
        """
        if k == 0:
            return 0.0
        
        # Get top k retrieved documents
        top_k_retrieved = retrieved_docs[:k]
        
        # Count relevant documents in top k
        relevant_in_top_k = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_docs)
        
        return relevant_in_top_k / k
    
    def calculate_recall_at_k(self, relevant_docs: List[str], retrieved_docs: List[str], k: int) -> float:
        """
        Calculate recall at k.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            k: Number of top results to consider
            
        Returns:
            Recall at k
        """
        if not relevant_docs:
            return 0.0
        
        # Get top k retrieved documents
        top_k_retrieved = retrieved_docs[:k]
        
        # Count relevant documents in top k
        relevant_in_top_k = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_docs)
        
        return relevant_in_top_k / len(relevant_docs)
    
    def calculate_mrr(self, relevant_docs: List[str], retrieved_docs: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            
        Returns:
            MRR score
        """
        if not relevant_docs:
            return 0.0
        
        reciprocal_ranks = []
        
        for relevant_doc in relevant_docs:
            try:
                rank = retrieved_docs.index(relevant_doc) + 1  # 1-indexed
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                # Document not found in retrieved list
                continue
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def calculate_ndcg_at_k(self, relevant_docs: List[str], retrieved_docs: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            relevance_scores: Dictionary mapping doc_id to relevance score
            k: Number of top results to consider
            
        Returns:
            NDCG at k
        """
        if k == 0:
            return 0.0
        
        # Get top k retrieved documents
        top_k_retrieved = retrieved_docs[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(top_k_retrieved):
            relevance = relevance_scores.get(doc_id, 0.0)
            dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = sorted([relevance_scores.get(doc_id, 0.0) for doc_id in relevant_docs], reverse=True)
        idcg = 0.0
        for i in range(min(k, len(ideal_relevance))):
            idcg += ideal_relevance[i] / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def benchmark_retrieval_system(self, pipeline, test_queries: List[Dict[str, Any]], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Benchmark a retrieval system with test queries.
        
        Args:
            pipeline: Retrieval pipeline to benchmark
            test_queries: List of test queries with ground truth
            k_values: List of k values for precision/recall calculation
            
        Returns:
            Benchmark results
        """
        with RetrievalLogger("Running retrieval system benchmark", logger):
            results = {
                'metrics': {},
                'latency': {},
                'per_query': []
            }
            
            # Initialize metrics
            for k in k_values:
                results['metrics'][f'precision_at_{k}'] = []
                results['metrics'][f'recall_at_{k}'] = []
                results['metrics'][f'ndcg_at_{k}'] = []
            
            results['metrics']['mrr'] = []
            results['latency']['retrieval_times'] = []
            results['latency']['reranking_times'] = []
            results['latency']['total_times'] = []
            
            # Process each test query
            for i, test_query in enumerate(test_queries):
                query = test_query['query']
                relevant_docs = test_query['relevant_docs']
                relevance_scores = test_query.get('relevance_scores', {})
                
                logger.info(f"Processing test query {i+1}/{len(test_queries)}: {query[:50]}...")
                
                # Measure retrieval time
                start_time = time.time()
                search_results = pipeline.search(query, top_k=max(k_values), enable_reranking=False)
                retrieval_time = time.time() - start_time
                
                # Measure reranking time
                start_time = time.time()
                reranked_results = pipeline.search(query, top_k=max(k_values), enable_reranking=True)
                reranking_time = time.time() - start_time
                
                total_time = retrieval_time + reranking_time
                
                # Extract document IDs
                retrieved_docs = [result.get('id', str(i)) for i, result in enumerate(search_results['results'])]
                reranked_docs = [result.get('id', str(i)) for i, result in enumerate(reranked_results['results'])]
                
                # Calculate metrics for different k values
                query_metrics = {
                    'query': query,
                    'retrieval_metrics': {},
                    'reranking_metrics': {},
                    'latency': {
                        'retrieval_time': retrieval_time,
                        'reranking_time': reranking_time,
                        'total_time': total_time
                    }
                }
                
                for k in k_values:
                    # Retrieval metrics
                    precision = self.calculate_precision_at_k(relevant_docs, retrieved_docs, k)
                    recall = self.calculate_recall_at_k(relevant_docs, retrieved_docs, k)
                    ndcg = self.calculate_ndcg_at_k(relevant_docs, retrieved_docs, relevance_scores, k)
                    
                    results['metrics'][f'precision_at_{k}'].append(precision)
                    results['metrics'][f'recall_at_{k}'].append(recall)
                    results['metrics'][f'ndcg_at_{k}'].append(ndcg)
                    
                    query_metrics['retrieval_metrics'][f'precision_at_{k}'] = precision
                    query_metrics['retrieval_metrics'][f'recall_at_{k}'] = recall
                    query_metrics['retrieval_metrics'][f'ndcg_at_{k}'] = ndcg
                    
                    # Reranking metrics
                    precision_rerank = self.calculate_precision_at_k(relevant_docs, reranked_docs, k)
                    recall_rerank = self.calculate_recall_at_k(relevant_docs, reranked_docs, k)
                    ndcg_rerank = self.calculate_ndcg_at_k(relevant_docs, reranked_docs, relevance_scores, k)
                    
                    query_metrics['reranking_metrics'][f'precision_at_{k}'] = precision_rerank
                    query_metrics['reranking_metrics'][f'recall_at_{k}'] = recall_rerank
                    query_metrics['reranking_metrics'][f'ndcg_at_{k}'] = ndcg_rerank
                
                # Calculate MRR
                mrr_retrieval = self.calculate_mrr(relevant_docs, retrieved_docs)
                mrr_reranking = self.calculate_mrr(relevant_docs, reranked_docs)
                
                results['metrics']['mrr'].append(mrr_reranking)
                query_metrics['retrieval_metrics']['mrr'] = mrr_retrieval
                query_metrics['reranking_metrics']['mrr'] = mrr_reranking
                
                # Store latency
                results['latency']['retrieval_times'].append(retrieval_time)
                results['latency']['reranking_times'].append(reranking_time)
                results['latency']['total_times'].append(total_time)
                
                results['per_query'].append(query_metrics)
            
            # Calculate averages (avoid modifying dict during iteration)
            metric_keys = list(results['metrics'].keys())
            for metric_name in metric_keys:
                values = results['metrics'][metric_name]
                results['metrics'][f'{metric_name}_avg'] = np.mean(values)
                results['metrics'][f'{metric_name}_std'] = np.std(values)
            
            # Calculate latency statistics (avoid modifying dict during iteration)
            latency_keys = list(results['latency'].keys())
            for latency_type in latency_keys:
                times = results['latency'][latency_type]
                results['latency'][f'{latency_type}_avg'] = np.mean(times)
                results['latency'][f'{latency_type}_std'] = np.std(times)
                results['latency'][f'{latency_type}_min'] = np.min(times)
                results['latency'][f'{latency_type}_max'] = np.max(times)
            
            return results
    
    def compare_rerankers(self, pipeline, test_queries: List[Dict[str, Any]], reranker_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare different reranker configurations.
        
        Args:
            pipeline: Retrieval pipeline
            test_queries: List of test queries
            reranker_configs: List of reranker configurations to test
            
        Returns:
            Comparison results
        """
        with RetrievalLogger("Comparing reranker configurations", logger):
            comparison_results = {
                'configurations': [],
                'overall_metrics': {}
            }
            
            for config in reranker_configs:
                logger.info(f"Testing configuration: {config['name']}")
                
                # Update pipeline configuration
                pipeline.update_config(config['config'])
                
                # Run benchmark
                config_results = self.benchmark_retrieval_system(pipeline, test_queries)
                
                comparison_results['configurations'].append({
                    'name': config['name'],
                    'config': config['config'],
                    'results': config_results
                })
            
            # Calculate overall comparison metrics
            for config_result in comparison_results['configurations']:
                config_name = config_result['name']
                metrics = config_result['results']['metrics']
                
                comparison_results['overall_metrics'][config_name] = {
                    'precision_at_5_avg': metrics['precision_at_5_avg'],
                    'recall_at_5_avg': metrics['recall_at_5_avg'],
                    'mrr_avg': metrics['mrr_avg'],
                    'avg_total_time': config_result['results']['latency']['total_times_avg']
                }
            
            return comparison_results
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """
        Save benchmark results to JSON file.
        
        Args:
            results: Benchmark results
            filename: Output filename
        """
        output_path = self.results_dir / filename
        
        with RetrievalLogger(f"Saving results to {output_path}", logger):
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {output_path}")
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable performance report.
        
        Args:
            results: Benchmark results
            
        Returns:
            Formatted performance report
        """
        report = []
        report.append("=" * 60)
        report.append("RETRIEVAL SYSTEM PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL METRICS:")
        report.append("-" * 20)
        metrics = results['metrics']
        
        for k in [1, 3, 5, 10]:
            if f'precision_at_{k}_avg' in metrics:
                report.append(f"Precision@{k}: {metrics[f'precision_at_{k}_avg']:.3f} ± {metrics[f'precision_at_{k}_std']:.3f}")
                report.append(f"Recall@{k}:    {metrics[f'recall_at_{k}_avg']:.3f} ± {metrics[f'recall_at_{k}_std']:.3f}")
                report.append(f"NDCG@{k}:      {metrics[f'ndcg_at_{k}_avg']:.3f} ± {metrics[f'ndcg_at_{k}_std']:.3f}")
                report.append("")
        
        if 'mrr_avg' in metrics:
            report.append(f"MRR:           {metrics['mrr_avg']:.3f} ± {metrics['mrr_std']:.3f}")
            report.append("")
        
        # Latency metrics
        report.append("LATENCY METRICS:")
        report.append("-" * 20)
        latency = results['latency']
        
        report.append(f"Retrieval Time:  {latency['retrieval_times_avg']:.3f}s ± {latency['retrieval_times_std']:.3f}s")
        report.append(f"Reranking Time:  {latency['reranking_times_avg']:.3f}s ± {latency['reranking_times_std']:.3f}s")
        report.append(f"Total Time:      {latency['total_times_avg']:.3f}s ± {latency['total_times_std']:.3f}s")
        report.append("")
        
        # Throughput
        total_queries = len(results['per_query'])
        total_time = sum(results['latency']['total_times'])
        throughput = total_queries / total_time if total_time > 0 else 0
        
        report.append(f"THROUGHPUT:")
        report.append("-" * 20)
        report.append(f"Queries per second: {throughput:.2f}")
        report.append(f"Total queries:      {total_queries}")
        report.append(f"Total time:         {total_time:.2f}s")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def create_test_queries() -> List[Dict[str, Any]]:
    """
    Create sample test queries for benchmarking.
    
    Returns:
        List of test queries with ground truth
    """
    return [
        {
            'query': 'How to create a webhook trigger for iFlow?',
            'relevant_docs': ['webhook_trigger_guide', 'trigger_setup', 'api_integration'],
            'relevance_scores': {
                'webhook_trigger_guide': 1.0,
                'trigger_setup': 0.9,
                'api_integration': 0.7
            }
        },
        {
            'query': 'Transform JSON data to XML format',
            'relevant_docs': ['json_xml_converter', 'data_transformation', 'format_conversion'],
            'relevance_scores': {
                'json_xml_converter': 1.0,
                'data_transformation': 0.8,
                'format_conversion': 0.6
            }
        },
        {
            'query': 'Handle database connection errors',
            'relevant_docs': ['error_handling', 'database_errors', 'connection_management'],
            'relevance_scores': {
                'error_handling': 0.9,
                'database_errors': 1.0,
                'connection_management': 0.8
            }
        },
        {
            'query': 'Map fields between different data schemas',
            'relevant_docs': ['field_mapping', 'schema_mapping', 'data_mapper'],
            'relevance_scores': {
                'field_mapping': 1.0,
                'schema_mapping': 0.9,
                'data_mapper': 0.8
            }
        },
        {
            'query': 'Aggregate data from multiple sources',
            'relevant_docs': ['data_aggregation', 'batch_processing', 'multi_source'],
            'relevance_scores': {
                'data_aggregation': 1.0,
                'batch_processing': 0.8,
                'multi_source': 0.7
            }
        }
    ]
