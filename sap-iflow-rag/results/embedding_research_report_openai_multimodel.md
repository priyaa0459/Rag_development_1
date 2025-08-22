# SAP iFlow Embedding Model Evaluation Report (OpenAI Multi-Model Strategy)
## Model Strategy
- **Primary**: OpenAI text-embedding-ada-002 (1536D)
- **Code Queries**: Microsoft CodeBERT (768D → padded to 1536D)
- **Fallback**: Sentence Transformers MiniLM (384D → padded to 1536D)

## Results
```json
{
  "text-embedding-ada-002": {
    "model_name": "text-embedding-ada-002",
    "embedding_time": 2.3527793884277344,
    "avg_time_per_text": 0.07842597961425782,
    "embedding_dimension": 1536,
    "sample_size": 30,
    "memory_usage_mb": 0.3515625,
    "avg_similarity": 0.0,
    "similarity_std": 0.0,
    "was_padded": false,
    "original_dimension": 1536,
    "clustering_score": 0.0
  },
  "microsoft/codebert-base": {
    "model_name": "microsoft/codebert-base",
    "embedding_time": 14.4404878616333,
    "avg_time_per_text": 0.4813495953877767,
    "embedding_dimension": 1536,
    "sample_size": 30,
    "memory_usage_mb": 0.17578125,
    "avg_similarity": 0.9891921281814575,
    "similarity_std": 0.005521870218217373,
    "was_padded": true,
    "original_dimension": 768,
    "clustering_score": 0.05286445841193199
  },
  "sentence-transformers/all-MiniLM-L6-v2": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_time": 2.025153636932373,
    "avg_time_per_text": 0.0675051212310791,
    "embedding_dimension": 1536,
    "sample_size": 30,
    "memory_usage_mb": 0.17578125,
    "avg_similarity": 0.5396106839179993,
    "similarity_std": 0.13736295700073242,
    "was_padded": true,
    "original_dimension": 1536,
    "clustering_score": 0.04660489037632942
  }
}```
## Recommendations
**Fastest model**: sentence-transformers/all-MiniLM-L6-v2 (0.068s/text)
**Most memory efficient**: microsoft/codebert-base (0.18MB)
**Best clustering**: microsoft/codebert-base (score 0.053)

**Multi-Model Strategy Recommendations**:
- **Primary**: Use OpenAI text-embedding-ada-002 for general queries
- **Code Queries**: Use CodeBERT for queries containing code-related keywords
- **Fallback**: Use MiniLM when OpenAI API is unavailable
- **Storage**: All embeddings padded to 1536D for consistency

## Model Comparison
| Model | Dim | Time/Text | Memory(MB) | Clust.Score | Padded |
|-------|-----|-----------|------------|-------------|--------|
| text-embedding-ada-002 | 1536 | 0.078s | 0.4 | 0.000 | No |
| microsoft/codebert-base | 1536 | 0.481s | 0.2 | 0.053 | Yes |
| sentence-transformers/all-MiniLM-L6-v2 | 1536 | 0.068s | 0.2 | 0.047 | Yes |


## Technical Details
- **Target Dimension**: 1536D for consistent storage
- **Padding Strategy**: Zero-padding for smaller embeddings
- **Model Selection**: Intelligent routing based on query content
- **Fallback Logic**: Graceful degradation on API failures


*Generated on 2025-08-22 11:08:54*
