# Retrieval and Reranking System

A comprehensive document retrieval and reranking system designed for iFlow components, featuring OpenAI-powered vector search, cross-encoder reranking, metadata-based scoring, and query classification with advanced benchmarking capabilities.

## Features

- **Vector Search**: Semantic search using OpenAI embeddings and FAISS
- **Cross-Encoder Reranking**: Advanced reranking using OpenAI GPT models
- **Metadata-based Reranking**: Scoring based on document complexity, reusability, freshness, and popularity
- **Enhanced Query Classification**: Dynamic switching between OpenAI, rule-based, and semantic approaches
- **Trainable Hybrid Reranker**: Machine learning-based fusion of multiple scoring signals
- **Comprehensive Benchmarking**: Precision@k, Recall@k, MRR, NDCG, and latency metrics
- **Component Suggestions**: Intelligent suggestions for relevant iFlow components
- **Batch Processing**: Efficient batch search and reranking capabilities
- **Persistence**: Save and load indices, models, and pipeline state
- **Performance Monitoring**: Detailed latency and throughput analysis

## Project Structure

```
retrieval_reranking/
├── src/
│   ├── utils/
│   │   ├── logging_utils.py          # Logging utilities
│   │   ├── query_utils.py            # Query processing with OpenAI
│   │   └── scoring_utils.py          # Score normalization and combination
│   ├── evaluation/
│   │   └── benchmarking.py           # Comprehensive benchmarking system
│   └── retrieval/
│       ├── vector_search.py          # OpenAI-powered vector search
│       ├── hybrid_search.py          # Hybrid search combining vector and keyword
│       ├── retrieval_pipeline.py     # Main pipeline orchestrator
│       └── rerankers/
│           ├── cross_encoder_reranker.py        # OpenAI cross-encoder reranking
│           ├── metadata_reranker.py             # Metadata-based reranking
│           ├── query_classifier.py              # Basic query classification
│           ├── enhanced_query_classifier.py     # Enhanced classification with dynamic switching
│           ├── hybrid_reranker.py               # Basic hybrid reranking
│           └── trainable_hybrid_reranker.py     # Trainable ML-based fusion
├── scripts/
│   ├── run_retrieval_demo.py         # Basic demo script
│   ├── end_to_end_demo.py            # Comprehensive end-to-end demo
│   └── run_benchmarks.py             # Complete benchmarking suite
├── results/                          # Benchmark results and performance reports
├── requirements.txt                  # Python dependencies
├── env.example                      # Environment configuration example
└── README.md                        # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd retrieval_reranking
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp env.example .env
   # Edit .env with your OpenAI API key and other settings
   ```

4. **Set your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   ```

## Quick Start

### Basic Usage

```python
from src.retrieval.retrieval_pipeline import RetrievalPipeline

# Initialize the pipeline
pipeline = RetrievalPipeline()

# Add sample documents
documents = [
    {
        'id': '1',
        'text': 'How to create a webhook trigger for iFlow',
        'metadata': {
            'complexity': 0.7,
            'reusability': 0.8,
            'freshness': 0.9,
            'popularity': 0.6
        }
    },
    # ... more documents
]

pipeline.add_documents(documents)

# Perform search
results = pipeline.search("webhook trigger setup", top_k=5)

# Get component suggestions
suggestions = pipeline.get_component_suggestions("webhook trigger setup")
```

### Enhanced Query Classification

```python
from src.retrieval.rerankers.enhanced_query_classifier import EnhancedQueryClassifier

# Initialize with adaptive strategy
classifier = EnhancedQueryClassifier(classification_strategy='adaptive')

# Classify query (automatically chooses best method)
classification = classifier.classify_query("how to create a webhook trigger")

# Switch strategies dynamically
classifier.set_classification_strategy('openai')
classification = classifier.classify_query("transform JSON to XML")

# Get component suggestions with method information
suggestions = classifier.get_component_suggestions("error handling")
for suggestion in suggestions:
    print(f"{suggestion['component_type']}: {suggestion['classification_method']}")
```

### Trainable Hybrid Reranker

```python
from src.retrieval.rerankers.trainable_hybrid_reranker import TrainableHybridReranker

# Initialize trainable reranker
reranker = TrainableHybridReranker(fusion_method='logistic_regression')

# Train the model
training_data = [
    {
        'vector_score': 0.8,
        'cross_encoder_score': 0.9,
        'metadata_score': 0.7,
        'classification_score': 0.6,
        'relevance_score': 1.0  # Ground truth
    },
    # ... more training examples
]

training_results = reranker.train(training_data, epochs=100)

# Use trained model for reranking
reranked = reranker.rerank(
    query="webhook trigger",
    documents=documents,
    vector_scores=[0.8, 0.6, 0.9],
    cross_encoder_scores=[0.9, 0.7, 0.8],
    metadata_scores=[0.7, 0.8, 0.6],
    classification_scores=[0.6, 0.5, 0.7]
)
```

## Configuration

The system uses environment variables for configuration. Key settings include:

### Model Configuration
- `VECTOR_MODEL_NAME`: OpenAI embedding model (default: `text-embedding-ada-002`)
- `CROSS_ENCODER_MODEL_NAME`: OpenAI model for reranking (default: `gpt-3.5-turbo`)
- `EMBEDDING_DIMENSION`: Embedding dimension (default: `1536`)

### Search Configuration
- `DEFAULT_TOP_K`: Default number of results (default: `10`)
- `DEFAULT_THRESHOLD`: Minimum similarity threshold (default: `0.0`)
- `ENABLE_RERANKING`: Whether to enable reranking (default: `true`)

### Weights Configuration
- `VECTOR_WEIGHT`: Weight for vector similarity (default: `0.6`)
- `CROSS_ENCODER_WEIGHT`: Weight for cross-encoder scores (default: `0.3`)
- `METADATA_WEIGHT`: Weight for metadata scores (default: `0.2`)

## Advanced Features

### 1. Vector Search with OpenAI Embeddings

```python
from src.retrieval.vector_search import VectorSearch

vector_search = VectorSearch(
    model_name='text-embedding-ada-002',
    dimension=1536
)

# Add documents
vector_search.add_documents(documents)

# Search
results = vector_search.search("query", top_k=10)

# Compute similarity
similarity = vector_search.compute_similarity("text1", "text2")
```

### 2. Cross-Encoder Reranking

```python
from src.retrieval.rerankers.cross_encoder_reranker import CrossEncoderReranker

reranker = CrossEncoderReranker(model_name='gpt-3.5-turbo')

# Rerank documents
reranked = reranker.rerank("query", documents, top_k=5)

# Get single document score
score = reranker.get_document_score("query", document)
```

### 3. Enhanced Query Classification

```python
from src.retrieval.rerankers.enhanced_query_classifier import EnhancedQueryClassifier

classifier = EnhancedQueryClassifier(classification_strategy='adaptive')

# Available strategies: 'openai', 'rule_based', 'semantic', 'adaptive'
classifier.set_classification_strategy('openai')

# Classify query
classification = classifier.classify_query("how to create a webhook trigger")

# Get component suggestions
suggestions = classifier.get_component_suggestions("webhook trigger")
```

### 4. Metadata-based Reranking

```python
from src.retrieval.rerankers.metadata_reranker import MetadataReranker

metadata_reranker = MetadataReranker()

# Rerank based on metadata
reranked = metadata_reranker.rerank("query", documents)
```

### 5. Trainable Hybrid Reranker

```python
from src.retrieval.rerankers.trainable_hybrid_reranker import TrainableHybridReranker

# Available fusion methods: 'logistic_regression', 'linear_regression', 'neural_network', 'weighted_sum'
reranker = TrainableHybridReranker(fusion_method='logistic_regression')

# Train the model
training_results = reranker.train(training_data, epochs=100)

# Use for reranking
reranked = reranker.rerank(query, documents, vector_scores, cross_encoder_scores, 
                          metadata_scores, classification_scores)
```

## Benchmarking and Evaluation

### Running Comprehensive Benchmarks

```bash
# Run all benchmarks
python scripts/run_benchmarks.py
```

This will generate:
- `results/comprehensive_benchmark.json`: System performance metrics
- `results/latency_benchmark.json`: Latency and throughput analysis
- `results/reranker_comparison.json`: Comparison of different reranker configurations
- `results/performance_report.json`: Final comprehensive report

### Manual Benchmarking

```python
from src.evaluation.benchmarking import RetrievalBenchmarker

# Create benchmarker
benchmarker = RetrievalBenchmarker()

# Run system benchmark
results = benchmarker.benchmark_retrieval_system(pipeline, test_queries)

# Generate performance report
report = benchmarker.generate_performance_report(results)
print(report)

# Save results
benchmarker.save_results(results, "my_benchmark.json")
```

### Metrics Calculated

- **Precision@k**: Precision at different k values (1, 3, 5, 10)
- **Recall@k**: Recall at different k values
- **MRR**: Mean Reciprocal Rank
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **Latency**: Retrieval and reranking times
- **Throughput**: Queries per second

## Demo Scripts

### Basic Demo

```bash
python scripts/run_retrieval_demo.py
```

### End-to-End Demo

```bash
python scripts/end_to_end_demo.py
```

Features:
- Interactive query testing
- Benchmark demo
- Component suggestions
- Score breakdown analysis
- Pipeline statistics

## Reranker Comparison

The system supports multiple reranker configurations:

| Configuration | Vector | Cross-Encoder | Metadata | Classification | Hybrid |
|---------------|--------|---------------|----------|----------------|--------|
| Vector_Only | ✓ | ✗ | ✗ | ✗ | ✗ |
| Vector_CrossEncoder | ✓ | ✓ | ✗ | ✗ | ✗ |
| Vector_Metadata | ✓ | ✗ | ✓ | ✗ | ✗ |
| Vector_Classification | ✓ | ✗ | ✗ | ✓ | ✗ |
| Full_Hybrid | ✓ | ✓ | ✓ | ✓ | ✓ |

### Performance Characteristics

- **Vector_Only**: Fastest, basic semantic search
- **Vector_CrossEncoder**: Good balance of speed and accuracy
- **Vector_Metadata**: Domain-specific optimization
- **Vector_Classification**: Component-aware ranking
- **Full_Hybrid**: Best accuracy, highest latency

## Evaluation

The system provides comprehensive evaluation metrics:

```python
# Get pipeline statistics
stats = pipeline.get_pipeline_stats()

# Get component-specific stats
vector_stats = pipeline.vector_search.get_index_stats()
classifier_stats = pipeline.query_classifier.get_classification_stats()
reranker_stats = pipeline.hybrid_reranker.get_reranker_stats()
```

## Customization

### Adding Custom Rerankers

```python
class CustomReranker:
    def rerank(self, query: str, documents: List[Dict], **kwargs):
        # Custom reranking logic
        return reranked_documents

# Use in hybrid reranker
hybrid_reranker = HybridReranker(
    custom_reranker=CustomReranker()
)
```

### Custom Scoring Functions

```python
from src.utils.scoring_utils import calculate_hybrid_score

# Custom hybrid scoring
score = calculate_hybrid_score(
    vector_score=0.8,
    keyword_score=0.6,
    metadata_score=0.7,
    cross_encoder_score=0.9,
    weights={'vector': 0.3, 'keyword': 0.2, 'metadata': 0.2, 'cross_encoder': 0.3}
)
```

### Training Custom Fusion Models

```python
# Create training data
training_data = [
    {
        'vector_score': 0.8,
        'cross_encoder_score': 0.9,
        'metadata_score': 0.7,
        'classification_score': 0.6,
        'relevance_score': 1.0  # Ground truth
    }
]

# Train model
reranker = TrainableHybridReranker(fusion_method='logistic_regression')
results = reranker.train(training_data, epochs=100)
```

## Logging

The system includes comprehensive logging:

```python
from src.utils.logging_utils import setup_logger, RetrievalLogger

# Setup logging
logger = setup_logger("my_app", log_file="app.log")

# Use context manager for timing
with RetrievalLogger("search operation", logger):
    results = pipeline.search("query")
```

## Performance Optimization

### Caching

```python
# Use cached cross-encoder reranker
from src.retrieval.rerankers.cross_encoder_reranker import CrossEncoderRerankerWithCache

reranker = CrossEncoderRerankerWithCache()
stats = reranker.get_cache_stats()
```

### Batch Processing

```python
# Batch search for multiple queries
queries = ["query1", "query2", "query3"]
batch_results = pipeline.batch_search(queries, top_k=5)
```

### Model Persistence

```python
# Save pipeline state
pipeline.save_pipeline()

# Load pipeline state
pipeline.load_pipeline()
```

## Troubleshooting

### Common Issues

1. **OpenAI API Errors**: Ensure your API key is set correctly
2. **Memory Issues**: Reduce batch sizes for large document collections
3. **Performance**: Use caching for frequently accessed data
4. **Model Loading**: Check file paths and permissions

### Performance Optimization

- Use `CrossEncoderRerankerWithCache` for better performance
- Adjust batch sizes based on your hardware
- Consider using smaller models for faster inference
- Enable caching for repeated queries

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
pipeline = RetrievalPipeline(log_file="debug.log")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- OpenAI for providing the embedding and language models
- FAISS for efficient similarity search
- The open-source community for various utilities and tools

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the demo scripts
- Run benchmarks to identify performance issues
