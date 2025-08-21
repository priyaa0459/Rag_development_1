"""
SAP iFlow Embedding Model Evaluation Framework (Updated for Cohere)
Person 1: Vector Storage and Embedding Research

Comprehensive evaluation of embedding models including Cohere for SAP iFlow code retrieval.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

class SAP_iFlowEmbeddingEvaluator:
    """Evaluate embedding models specifically for SAP iFlow code generation tasks."""
    
    def __init__(self):
        self.test_queries = [
            # Invoice Processing Queries
            "Create an invoice request integration flow between supplier and S4HANA Cloud",
            "Configure SOAP adapter for invoice processing with authentication",
            "Set up email notifications for invoice processing errors",
            "Map EDI INVOIC format to S4HANA InvoiceRequest structure",
            
            # Business Partner Queries  
            "Business partner relationship replication from S4HANA to CRM",
            "Configure endpoint for business partner data synchronization",
            "Create message mapping for business partner confirmation",
            
            # General Integration Queries
            "HTTPS sender adapter configuration with certificate authentication", 
            "Create Groovy script for message processing and logging",
            "Configure timeout settings for SOAP service calls",
            "Set up parameterized properties for different environments",
            
            # Technical Configuration Queries
            "Create Eclipse project structure for SAP Cloud Integration",
            "Configure WSDL service definition with SOAP bindings",
            "Set up error handling with retry mechanisms"
        ]
        
        self.evaluation_metrics = {}
        
        # Query categories for evaluation
        self.query_categories = [
            'invoice_processing', 'invoice_processing', 'invoice_processing', 'invoice_processing',
            'business_partner', 'business_partner', 'business_partner',
            'integration_config', 'integration_config', 'integration_config', 'integration_config',
            'technical_config', 'technical_config', 'technical_config'
        ]
    
    def evaluate_retrieval_accuracy(self, 
                                   embeddings: np.ndarray, 
                                   texts: List[str], 
                                   categories: List[str],
                                   query_embeddings: np.ndarray,
                                   query_categories: List[str]) -> Dict:
        """Evaluate how well the embedding model retrieves relevant SAP iFlow content."""
        
        results = {
            'top1_accuracy': 0,
            'top3_accuracy': 0, 
            'top5_accuracy': 0,
            'mrr': 0,  # Mean Reciprocal Rank
            'category_precision': {}
        }
        
        total_queries = len(query_embeddings)
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        reciprocal_ranks = []
        
        for i, (query_emb, query_cat) in enumerate(zip(query_embeddings, query_categories)):
            # Calculate similarities
            similarities = cosine_similarity([query_emb], embeddings)[0]
            
            # Get top 5 most similar items
            top_indices = np.argsort(similarities)[::-1][:5]
            top_categories = [categories[idx] for idx in top_indices]
            
            # Check if query category appears in top results
            relevant_positions = [j for j, cat in enumerate(top_categories) if cat == query_cat]
            
            if relevant_positions:
                top1_correct += 1 if 0 in relevant_positions else 0
                top3_correct += 1 if any(pos < 3 for pos in relevant_positions) else 0
                top5_correct += 1 if any(pos < 5 for pos in relevant_positions) else 0
                
                # Calculate reciprocal rank
                first_relevant_pos = min(relevant_positions) + 1
                reciprocal_ranks.append(1.0 / first_relevant_pos)
            else:
                reciprocal_ranks.append(0.0)
        
        results['top1_accuracy'] = top1_correct / total_queries
        results['top3_accuracy'] = top3_correct / total_queries
        results['top5_accuracy'] = top5_correct / total_queries
        results['mrr'] = np.mean(reciprocal_ranks)
        
        return results
    
    def evaluate_semantic_clustering(self, embeddings: np.ndarray, labels: List[str]) -> Dict:
        """Evaluate how well embeddings cluster similar SAP iFlow components."""
        
        # Create numeric labels
        unique_labels = list(set(labels))
        label_to_num = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = [label_to_num[label] for label in labels]
        
        clustering_results = {}
        
        # Silhouette score
        if len(unique_labels) > 1:
            sil_score = silhouette_score(embeddings, numeric_labels)
            clustering_results['silhouette_score'] = sil_score
        
        # Intra-cluster vs inter-cluster similarity
        intra_similarities = []
        inter_similarities = []
        
        for i, label_i in enumerate(labels):
            for j, label_j in enumerate(labels):
                if i != j:
                    similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    if label_i == label_j:
                        intra_similarities.append(similarity)
                    else:
                        inter_similarities.append(similarity)
        
        clustering_results['avg_intra_similarity'] = np.mean(intra_similarities) if intra_similarities else 0
        clustering_results['avg_inter_similarity'] = np.mean(inter_similarities) if inter_similarities else 0
        clustering_results['separation_ratio'] = (
            clustering_results['avg_intra_similarity'] / clustering_results['avg_inter_similarity'] 
            if clustering_results['avg_inter_similarity'] > 0 else 0
        )
        
        return clustering_results
    
    def evaluate_cohere_specific_metrics(self, cohere_embeddings: np.ndarray, 
                                       local_embeddings: np.ndarray,
                                       texts: List[str]) -> Dict:
        """Evaluate Cohere-specific performance compared to local models."""
        
        cohere_results = {}
        
        # Embedding quality comparison
        if cohere_embeddings.shape[0] == local_embeddings.shape[0]:
            # Calculate correlation between embedding spaces
            cohere_flat = cohere_embeddings.flatten()
            local_flat = local_embeddings.flatten()
            
            # Sample for correlation (too many points for full correlation)
            sample_size = min(10000, len(cohere_flat))
            indices = np.random.choice(len(cohere_flat), sample_size, replace=False)
            
            correlation = np.corrcoef(cohere_flat[indices], local_flat[indices])[0, 1]
            cohere_results['embedding_space_correlation'] = correlation
        
        # Semantic richness (higher dimensional space)
        cohere_results['embedding_dimension'] = cohere_embeddings.shape[1]
        cohere_results['local_dimension'] = local_embeddings.shape[1]
        cohere_results['dimension_ratio'] = cohere_embeddings.shape[1] / local_embeddings.shape[1]
        
        # Variance in embedding space (indicates semantic richness)
        cohere_results['cohere_variance'] = np.var(cohere_embeddings, axis=0).mean()
        cohere_results['local_variance'] = np.var(local_embeddings, axis=0).mean()
        
        return cohere_results
    
    def evaluate_code_type_specificity(self, embeddings: np.ndarray, output_types: List[str]) -> Dict:
        """Evaluate how well embeddings distinguish between different types of SAP iFlow code."""
        
        type_results = {}
        
        # Get unique output types
        unique_types = list(set(output_types))
        
        for output_type in unique_types:
            # Get embeddings for this type
            type_indices = [i for i, t in enumerate(output_types) if t == output_type]
            
            if len(type_indices) < 2:
                continue
                
            type_embeddings = embeddings[type_indices]
            
            # Calculate average similarity within this type
            similarities = cosine_similarity(type_embeddings)
            avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
            
            type_results[output_type] = {
                'count': len(type_indices),
                'avg_intra_similarity': avg_similarity
            }
        
        return type_results
    
    def benchmark_query_performance(self, 
                                   model_embeddings: Dict[str, np.ndarray],
                                   texts: List[str],
                                   categories: List[str]) -> Dict:
        """Benchmark different models on SAP iFlow-specific queries."""
        
        benchmark_results = {}
        
        for model_name, embeddings in model_embeddings.items():
            print(f"Benchmarking {model_name}...")
            
            model_results = {
                'retrieval_accuracy': {},
                'semantic_clustering': {},
                'code_type_specificity': {},
                'model_info': {
                    'dimension': embeddings.shape[1],
                    'type': 'cohere' if 'cohere' in model_name.lower() else 'local'
                }
            }
            
            # Test semantic clustering
            clustering_results = self.evaluate_semantic_clustering(embeddings, categories)
            model_results['semantic_clustering'] = clustering_results
            
            # Test code type specificity
            type_results = self.evaluate_code_type_specificity(embeddings, categories)
            model_results['code_type_specificity'] = type_results
            
            benchmark_results[model_name] = model_results
        
        return benchmark_results
    
    def generate_cohere_comparison_report(self, results: Dict, cohere_cost_info: Dict = None) -> str:
        """Generate a comprehensive evaluation report comparing Cohere with local models."""
        
        report = """# SAP iFlow Embedding Model Evaluation Report (with Cohere)

## Executive Summary

This report evaluates embedding models for the SAP iFlow RAG pipeline, comparing:
- **Cohere embed-english-v3** (1024D, API-based)
- **Cohere embed-multilingual-v3** (1024D, API-based)  
- **Sentence Transformers** (384D, local)
- **CodeBERT** (768D, local)

Focus areas: SAP-specific code retrieval, semantic clustering, and cost-effectiveness.

## Evaluation Methodology

### Test Queries
SAP iFlow-specific queries across categories:
- Invoice processing integration flows (4 queries)
- Business partner replication scenarios (3 queries)  
- Technical adapter configurations (4 queries)
- Message mapping and transformations (3 queries)

### Metrics
- **Retrieval Accuracy**: Top-K accuracy for relevant code retrieval
- **Semantic Clustering**: Silhouette score and separation ratio
- **Code Type Specificity**: Intra-type similarity for SAP components
- **Performance**: Speed, memory usage, and cost analysis

## Results Summary

"""
        
        # Add detailed results for each model
        for model_name, model_results in results.items():
            is_cohere = 'cohere' in model_name.lower()
            
            report += f"### {model_name}\n"
            report += f"**Type**: {'API-based (Cohere)' if is_cohere else 'Local model'}\n"
            
            if 'model_info' in model_results:
                info = model_results['model_info']
                report += f"**Dimension**: {info['dimension']}D\n"
            
            if 'semantic_clustering' in model_results:
                clustering = model_results['semantic_clustering']
                if 'silhouette_score' in clustering:
                    report += f"**Silhouette Score**: {clustering['silhouette_score']:.3f}\n"
                if 'separation_ratio' in clustering:
                    report += f"**Separation Ratio**: {clustering['separation_ratio']:.3f}\n"
                if 'avg_intra_similarity' in clustering:
                    report += f"**Intra-cluster Similarity**: {clustering['avg_intra_similarity']:.3f}\n"
            
            if 'code_type_specificity' in model_results:
                type_results = model_results['code_type_specificity']
                report += f"**Code Types Recognized**: {len(type_results)}\n"
                
                # Show top performing code types
                sorted_types = sorted(type_results.items(), 
                                    key=lambda x: x[1]['avg_intra_similarity'], 
                                    reverse=True)[:3]
                
                for code_type, stats in sorted_types:
                    report += f"  - {code_type}: {stats['avg_intra_similarity']:.3f} similarity ({stats['count']} samples)\n"
            
            report += "\n"
        
        # Cost analysis section
        if cohere_cost_info:
            report += """## Cost Analysis

### Cohere API Costs
"""
            for key, value in cohere_cost_info.items():
                report += f"- **{key}**: {value}\n"
            
            report += """
### Cost Comparison
- **Local Models**: One-time setup cost, ongoing compute resources
- **Cohere API**: Per-embedding pricing, no infrastructure management
- **Hybrid Approach**: Cohere for queries, local for bulk document processing
"""
        
        report += """## Recommendations

### Production Deployment Strategy:

#### Primary Recommendation: Hybrid Approach
1. **Query Processing**: Use **Cohere embed-english-v3** for user queries
   - Higher dimensional embeddings (1024D) for better semantic understanding
   - Optimized for search queries with `input_type='search_query'`
   - Better handling of natural language questions

2. **Document Storage**: Use **all-MiniLM-L6-v2** for bulk SAP iFlow embedding
   - Cost-effective for large document collections
   - Fast local processing for batch operations
   - Sufficient quality for document representation

#### Alternative: Pure Cohere Approach
- Use Cohere for both queries and documents if budget allows
- Best semantic quality and consistency
- Requires API rate limit management

#### Alternative: Pure Local Approach  
- Use all-MiniLM-L6-v2 or all-mpnet-base-v2 for everything
- No API dependencies or costs
- Good performance for technical content

### Implementation Considerations:

#### For Cohere Integration:
- Implement proper rate limiting (96 texts per batch)
- Add retry logic for API failures
- Monitor usage and costs
- Cache embeddings to avoid re-computation

#### For Vector Database:
- Use variable-dimension schema to support both model types
- Index by embedding model for filtered searches
- Implement fallback mechanisms

#### Performance Optimization:
- Batch embed documents during off-peak hours
- Use Cohere for real-time query processing
- Implement semantic caching for frequent queries

### Next Steps:
1. **Implement hybrid embedding strategy**
2. **Set up cost monitoring for Cohere API**
3. **Create fallback mechanisms for API failures**
4. **Monitor retrieval performance in production**
5. **Fine-tune similarity thresholds based on user feedback**

## Technical Implementation

### Database Schema Updates:
```sql
-- Support for multiple embedding dimensions
ALTER TABLE sap_iflow_chunks 
ADD COLUMN embedding_model VARCHAR(100),
ADD COLUMN embedding_dimension INTEGER;

-- Create separate similarity search functions for different models
CREATE OR REPLACE FUNCTION search_with_cohere_embeddings(...)
CREATE OR REPLACE FUNCTION search_with_local_embeddings(...)
```

### API Integration Pattern:
```python
async def get_embedding(text, is_query=False):
    if is_query and cohere_available():
        return await cohere_embed(text, input_type='search_query')
    else:
        return local_model.encode([text])[0]
```

### Monitoring Setup:
- Track Cohere API usage and costs
- Monitor embedding generation latency  
- Log retrieval accuracy metrics
- Set up alerts for API rate limits
"""
        
        return report
    
    def visualize_model_comparison(self, 
                                 model_embeddings: Dict[str, np.ndarray], 
                                 labels: List[str],
                                 save_path: str = "model_comparison.png"):
        """Create visualizations comparing different embedding models."""
        
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        fig, axes = plt.subplots(2, len(model_embeddings), figsize=(5*len(model_embeddings), 10))
        if len(model_embeddings) == 1:
            axes = axes.reshape(-1, 1)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(set(labels))))
        unique_labels = list(set(labels))
        
        for idx, (model_name, embeddings) in enumerate(model_embeddings.items()):
            # PCA visualization
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d_pca = pca.fit_transform(embeddings)
            
            ax_pca = axes[0, idx]
            for label_idx, label in enumerate(unique_labels):
                mask = [l == label for l in labels]
                ax_pca.scatter(embeddings_2d_pca[mask, 0], embeddings_2d_pca[mask, 1], 
                             c=[colors[label_idx]], label=label, alpha=0.7)
            
            ax_pca.set_title(f'{model_name} - PCA\n({embeddings.shape[1]}D → 2D)')
            ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            if idx == 0:
                ax_pca.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # t-SNE visualization  
            if embeddings.shape[0] > 10:  # Only if we have enough samples
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0]-1))
                embeddings_2d_tsne = tsne.fit_transform(embeddings)
                
                ax_tsne = axes[1, idx]
                for label_idx, label in enumerate(unique_labels):
                    mask = [l == label for l in labels]
                    ax_tsne.scatter(embeddings_2d_tsne[mask, 0], embeddings_2d_tsne[mask, 1],
                                  c=[colors[label_idx]], label=label, alpha=0.7)
                
                ax_tsne.set_title(f'{model_name} - t-SNE\n({embeddings.shape[1]}D → 2D)')
                ax_tsne.set_xlabel('t-SNE 1')
                ax_tsne.set_ylabel('t-SNE 2')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

def main():
    """Main evaluation function."""
    print("SAP iFlow Embedding Model Evaluation Framework (with Cohere)")
    print("=" * 70)
    
    print("Evaluation framework ready for Cohere integration.")
    print("Features:")
    print("- Compare Cohere vs local embedding models")
    print("- Evaluate semantic clustering quality")
    print("- Analyze cost vs performance trade-offs")
    print("- Generate comprehensive comparison reports")
    print("\nRun this after generating embeddings with both Cohere and local models.")

if __name__ == "__main__":
    main()