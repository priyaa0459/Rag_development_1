# SAP iFlow RAG Pipeline - Person 1: Vector Storage and Embedding Research

## Project Overview
This module handles embedding model research, vector database setup, and loading SAP iFlow training chunks into Supabase using OpenAI's multi-model strategy.

## Multi-Model Embedding Strategy

### **Primary Model**: OpenAI text-embedding-ada-002 (1536D)
- High-quality embeddings for general queries
- Optimized for natural language understanding
- API-based with rate limiting and fallback

### **Code-Specific Model**: Microsoft CodeBERT (768D → padded to 1536D)
- Automatically selected for queries containing code keywords
- Keywords: code, script, function, method, class, variable, parameter, api, endpoint
- Padded to 1536D for consistent storage

### **Fallback Model**: Sentence Transformers MiniLM (384D → padded to 1536D)
- Used when OpenAI API is unavailable
- Fast local processing
- Padded to 1536D for consistent storage

## Setup

1. Create and activate Python environment:

2. Install dependencies:

3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   SUPABASE_URL=your_supabase_url
   SUPABASE_ANON_KEY=your_supabase_anon_key
   ```

## Workflow

- Run `embedding_research_openai.py` to benchmark all three embedding models
- Execute schema creation and load data with `vector_loader_openai.py`
- Test intelligent model selection with `test_soap_query.py`

## Technical Features

- **Intelligent Model Selection**: Automatically routes queries to the best model
- **Dimension Consistency**: All embeddings stored as 1536D vectors
- **Smart Fallback**: Graceful degradation when primary models fail
- **Metadata Tracking**: Records which model generated each embedding
- **Batch Processing**: Efficient handling of large datasets
- **Rate Limiting**: Respectful API usage with delays

## Project Structure

Describe main folders and files.


