# SAP iFlow RAG (Persons 1-3) – Backend API for UI Integration (Person 4)

This repository provides the Retrieval-Augmented Generation (RAG) backend for SAP iFlow code/config generation. It includes:
- Person 1: Embedding research and evaluation utilities
- Person 2: Retrieval and re-ranking via Supabase vectors + cross-encoder
- Person 3: RAG code/config generation via Cohere

## What’s New
- FastAPI REST layer exposing `/search` and `/generate`
- Structured JSON suited for UI (Person 4)
- Mock mode for local testing without Supabase/LLM
- Robust error handling and validation

## Quick Start

### 1) Create and activate virtual environment (optional)
You already have a `sap-iflow-rag` venv folder in the repo. You can use it or create your own.

Windows PowerShell (recommended):
```bash
%CD%\\sap-iflow-rag\\Scripts\\Activate.ps1
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure environment
Create a `.env` file in the project root based on `.env.example`:
```bash
copy .env.example .env
```
Then set:
- `SUPABASE_URL` and `SUPABASE_ANON_KEY` (for retrieval)
- `COHERE_API_KEY` (for generation)
- `MOCK_MODE=true` to run locally without external services

### 4) Run the API server
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Open docs at: `http://localhost:8000/docs`

## REST Endpoints

### GET /health
Simple liveness probe.

### POST /search
Body:
```json
{
  "query": "Create invoice request iFlow",
  "top_k": 5,
  "match_threshold": 0.5,
  "mock": true
}
```

Response (truncated example):
```json
{
  "ok": true,
  "query": "Create invoice request iFlow",
  "retrieval_metadata": {"top_k":5, "match_threshold":0.5, "mode":"mock"},
  "results": [
    {
      "chunk_id": "demo_chunk_1",
      "instruction": "...",
      "input_context": "...",
      "output_code": "...",
      "output_type": "XML",
      "embedding_model": "codebert-base",
      "codebert_similarity": 0.72,
      "cross_encoder_score": 0.63,
      "hybrid_score": 0.69,
      "metadata": {"data_type":"...","source_file":"..."}
    }
  ]
}
```

### POST /generate
Body (server performs search unless `contexts` provided):
```json
{
  "query": "Create invoice request iFlow",
  "output_type": "xml",
  "top_k": 5,
  "temperature": 0.3,
  "max_tokens": 2048,
  "mock": true
}
```

Response (truncated example):
```json
{
  "ok": true,
  "query": "Create invoice request iFlow",
  "output_type": "xml",
  "retrieval_metadata": {"mode":"mock","top_k":5},
  "retrieval_context": [ { "chunk_id": "..." } ],
  "generated": {
    "query": "Create invoice request iFlow",
    "output_type": "xml",
    "generated_code": "<iflow>...</iflow>",
    "validation_status": "VALID_XML",
    "confidence_score": 0.81,
    "context_chunks_used": ["chunk_id: ..."],
    "generation_metadata": {
      "model": "cohere-command",
      "max_tokens": 2048,
      "temperature": 0.3,
      "prompt_length": 1234,
      "generation_id": "gen_..."
    }
  }
}
```

You can also pass your own contexts to bypass retrieval:
```json
{
  "query": "Create invoice request iFlow",
  "output_type": "groovy",
  "contexts": [
    {
      "chunk_id": "c1",
      "instruction": "...",
      "input_context": "...",
      "output_code": "...",
      "output_type": "XML",
      "embedding_model": "codebert-base",
      "codebert_similarity": 0.8,
      "cross_encoder_score": 0.0,
      "hybrid_score": 0.8,
      "metadata": {"source":"ui"}
    }
  ]
}
```

## CLI / Local Testing
- Retrieval demos: `python src/retriever.py demo` or `python src/retriever.py interactive`
- Generation demos: `python src/generator.py demo` or `python src/generator.py interactive`

## Environment
See `.env.example` for all variables. Minimum:
- `MOCK_MODE=true` (for offline testing)
- `SUPABASE_URL`, `SUPABASE_ANON_KEY` (if using real retrieval)
- `COHERE_API_KEY` (if using real generation)

## Data
Sample processed CSV is available at `data/processed/processed_sap_iflow_data.csv` and used by mock retrieval when available.

## Error Handling
- Invalid requests return 400 with details
- Missing env or services returns actionable 400/500
- Timeouts or LLM errors return 500 with safe, structured payload including `validation_status` and `generation_metadata.error`

## Testing From UI/JS
Example fetch to `/generate`:
```javascript
const res = await fetch("http://localhost:8000/generate", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({ query: "Create invoice request iFlow", output_type: "xml", mock: true })
});
const data = await res.json();
```

## Notes
- Ensure Python 3.10+ and `pip` are installed.
- If using the included venv on Windows: `%CD%\\sap-iflow-rag\\Scripts\\Activate.ps1`
