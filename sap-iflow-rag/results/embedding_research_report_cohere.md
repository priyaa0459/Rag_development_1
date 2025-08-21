# SAP iFlow Embedding Model Evaluation Report (with Cohere)
## Results
```
{
  "all-MiniLM-L6-v2": {
    "model_name": "all-MiniLM-L6-v2",
    "embedding_time": 1.1921019554138184,
    "avg_time_per_text": 0.039736731847127275,
    "embedding_dimension": 384,
    "sample_size": 30,
    "memory_usage_mb": 0.0439453125,
    "avg_similarity": 0.5396106839179993,
    "similarity_std": 0.13736295700073242,
    "clustering_score": 0.04660489037632942
  },
  "all-mpnet-base-v2": {
    "model_name": "all-mpnet-base-v2",
    "embedding_time": 11.47001028060913,
    "avg_time_per_text": 0.3823336760203044,
    "embedding_dimension": 768,
    "sample_size": 30,
    "memory_usage_mb": 0.087890625,
    "avg_similarity": 0.5459361672401428,
    "similarity_std": 0.14646169543266296,
    "clustering_score": 0.003208050737157464
  },
  "codebert-base": {
    "model_name": "codebert-base",
    "embedding_time": 10.275477409362793,
    "avg_time_per_text": 0.34251591364542644,
    "embedding_dimension": 768,
    "sample_size": 30,
    "memory_usage_mb": 0.087890625,
    "avg_similarity": 0.9891921281814575,
    "similarity_std": 0.005521869752556086,
    "clustering_score": 0.05286448448896408
  },
  "cohere-embed-english-v3": {
    "model_name": "cohere-embed-english-v3",
    "embedding_time": 2.2102904319763184,
    "avg_time_per_text": 0.07367634773254395,
    "embedding_dimension": 1024,
    "sample_size": 30,
    "memory_usage_mb": 0.234375,
    "avg_similarity": 0.6179724301997196,
    "similarity_std": 0.1266414251116522,
    "clustering_score": 0.034815915597623905
  },
  "cohere-embed-multilingual-v3": {
    "model_name": "cohere-embed-multilingual-v3",
    "embedding_time": 1.3831510543823242,
    "avg_time_per_text": 0.04610503514607747,
    "embedding_dimension": 1024,
    "sample_size": 30,
    "memory_usage_mb": 0.234375,
    "avg_similarity": 0.6894237956984748,
    "similarity_std": 0.10358759451005953,
    "clustering_score": 0.029140356129284194
  }
}```
## Recommendations
**Fastest model**: all-MiniLM-L6-v2 (0.040s/text)
**Most memory efficient**: all-MiniLM-L6-v2 (0.04MB)
**Best clustering**: codebert-base (score 0.053)

**Recommendation**: Choose based on your priorities—speed, memory, or clustering quality.

## Model Comparison
| Model | Dim | Time/Text | Memory(MB) | Clust.Score |
|-------|-----|-----------|------------|-------------|
| all-MiniLM-L6-v2 | 384 | 0.040s | 0.0 | 0.047 |
| all-mpnet-base-v2 | 768 | 0.382s | 0.1 | 0.003 |
| codebert-base | 768 | 0.343s | 0.1 | 0.053 |
| cohere-embed-english-v3 | 1024 | 0.074s | 0.2 | 0.035 |
| cohere-embed-multilingual-v3 | 1024 | 0.046s | 0.2 | 0.029 |


*Generated on 2025-08-20 22:00:22*
