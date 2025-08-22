# SAP iFlow Embedding Model Evaluation Report (with OpenAI)
## Results
```
{
  "all-MiniLM-L6-v2": {
    "model_name": "all-MiniLM-L6-v2",
    "embedding_time": 1.4313757419586182,
    "avg_time_per_text": 0.04771252473195394,
    "embedding_dimension": 384,
    "sample_size": 30,
    "memory_usage_mb": 0.0439453125,
    "avg_similarity": 0.5396106839179993,
    "similarity_std": 0.13736295700073242,
    "clustering_score": 0.04660489037632942
  },
  "all-mpnet-base-v2": {
    "model_name": "all-mpnet-base-v2",
    "embedding_time": 9.933702945709229,
    "avg_time_per_text": 0.33112343152364093,
    "embedding_dimension": 768,
    "sample_size": 30,
    "memory_usage_mb": 0.087890625,
    "avg_similarity": 0.5459361672401428,
    "similarity_std": 0.14646169543266296,
    "clustering_score": 0.003208050737157464
  },
  "codebert-base": {
    "model_name": "codebert-base",
    "embedding_time": 11.148939847946167,
    "avg_time_per_text": 0.3716313282648722,
    "embedding_dimension": 768,
    "sample_size": 30,
    "memory_usage_mb": 0.087890625,
    "avg_similarity": 0.9891921281814575,
    "similarity_std": 0.005521869752556086,
    "clustering_score": 0.05286448448896408
  },
  "text-embedding-ada-002": {
    "model_name": "text-embedding-ada-002",
    "embedding_time": 0.013783454895019531,
    "avg_time_per_text": 0.00045944849650065103,
    "embedding_dimension": 1536,
    "sample_size": 30,
    "memory_usage_mb": 0.3515625,
    "avg_similarity": 0.0,
    "similarity_std": 0.0,
    "clustering_score": 0.0
  }
}```
## Recommendations
**Fastest model**: text-embedding-ada-002 (0.000s/text)
**Most memory efficient**: all-MiniLM-L6-v2 (0.04MB)
**Best clustering**: codebert-base (score 0.053)

**Recommendation**: Choose based on your priorities—speed, memory, or clustering quality.

## Model Comparison
| Model | Dim | Time/Text | Memory(MB) | Clust.Score |
|-------|-----|-----------|------------|-------------|
| all-MiniLM-L6-v2 | 384 | 0.048s | 0.0 | 0.047 |
| all-mpnet-base-v2 | 768 | 0.331s | 0.1 | 0.003 |
| codebert-base | 768 | 0.372s | 0.1 | 0.053 |
| text-embedding-ada-002 | 1536 | 0.000s | 0.4 | 0.000 |


*Generated on 2025-08-22 09:42:35*
