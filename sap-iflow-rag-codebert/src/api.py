# pyright: reportMissingImports=false
"""
FastAPI REST API for SAP iFlow RAG (Persons 1-3) exposing endpoints for UI/Person 4.

Endpoints:
- GET /health
- POST /search    → Person 2 retrieval and reranking
- POST /generate  → Person 3 RAG generation (uses /search results unless contexts provided)

The API returns structured JSON with original query, output type, metadata, retrieval context,
and generated code/config.
"""

import os
import traceback
from typing import List, Optional, Any, Dict, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

try:
    from src.retriever import SAPiFlowRetriever, SearchResult
except ImportError:
    from retriever import SAPiFlowRetriever, SearchResult

try:
    from src.generator import SAPiFlowGenerator, GenerationRequest
except ImportError:
    from generator import SAPiFlowGenerator, GenerationRequest

import pandas as pd


load_dotenv()

app = FastAPI(title="SAP iFlow RAG API", version="0.1.0")


# -----------------------------
# Pydantic Schemas
# -----------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    match_threshold: float = Field(0.5, ge=0.0, le=1.0)


class RetrievedContextChunk(BaseModel):
    chunk_id: str
    instruction: str
    input_context: str
    output_code: str
    output_type: str
    embedding_model: str
    codebert_similarity: float
    cross_encoder_score: float
    hybrid_score: float
    metadata: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    ok: bool
    query: str
    retrieval_metadata: Dict[str, Any]
    results: List[RetrievedContextChunk]


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    output_type: Literal["xml", "groovy", "properties", "integration_steps"] = "xml"
    top_k: int = Field(5, ge=1, le=20)
    temperature: float = Field(0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(2048, ge=64, le=4096)
    # Optional: allow UI to pass its own retrieval contexts (bypass /search)
    contexts: Optional[List[RetrievedContextChunk]] = None


class GenerateResponse(BaseModel):
    ok: bool
    query: str
    output_type: str
    retrieval_metadata: Dict[str, Any]
    retrieval_context: List[RetrievedContextChunk]
    generated: Dict[str, Any]


# -----------------------------
# Helpers
# -----------------------------

def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y"}


def _serialize_result(result: SearchResult) -> Dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "instruction": result.instruction,
        "input_context": result.input_context,
        "output_code": result.output_code,
        "output_type": result.output_type,
        "embedding_model": result.embedding_model,
        "codebert_similarity": float(result.codebert_similarity),
        "cross_encoder_score": float(result.cross_encoder_score),
        "hybrid_score": float(result.hybrid_score),
        "metadata": result.metadata or {},
    }


def _do_search(req: SearchRequest) -> SearchResponse:
    retrieval_meta: Dict[str, Any] = {
        "top_k": req.top_k,
        "match_threshold": req.match_threshold,
        "mode": "supabase",
    }

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    if not supabase_url or not supabase_key:
        raise HTTPException(status_code=400, detail="SUPABASE_URL and SUPABASE_ANON_KEY are not set. Enable MOCK_MODE or provide credentials.")

    retriever = SAPiFlowRetriever(supabase_url, supabase_key)
    retriever.initialize_models()

    results = retriever.search(req.query, top_k=req.top_k, match_threshold=req.match_threshold)
    serialized = [_serialize_result(r) for r in results]
    retrieval_meta.update(retriever.get_retrieval_stats() or {})
    return SearchResponse(ok=True, query=req.query, retrieval_metadata=retrieval_meta, results=serialized)


# -----------------------------
# Routes
# -----------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "SAP iFlow RAG API", "docs": "/docs"}


@app.get("/favicon.ico")
def favicon():
    return JSONResponse(status_code=204, content=None)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    try:
        return _do_search(req)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "query": req.query,
                "retrieval_metadata": {"error": str(e)},
                "results": [],
            },
        )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        # Determine retrieval context
        if req.contexts and len(req.contexts) > 0:
            contexts_serialized = [c.dict() for c in req.contexts]
            retrieval_meta = {"source": "client_provided", "count": len(contexts_serialized)}
        else:
            search_resp = _do_search(
                SearchRequest(
                    query=req.query,
                    top_k=req.top_k,
                    match_threshold=0.5,
                )
            )
            contexts_serialized = search_resp.results
            retrieval_meta = search_resp.retrieval_metadata

        # Convert to SearchResult objects for generator
        search_results: List[SearchResult] = []
        for c in contexts_serialized:
            search_results.append(
                SearchResult(
                    chunk_id=c.get("chunk_id", ""),
                    instruction=c.get("instruction", ""),
                    input_context=c.get("input_context", ""),
                    output_code=c.get("output_code", ""),
                    output_type=c.get("output_type", ""),
                    embedding_model=c.get("embedding_model", ""),
                    codebert_similarity=float(c.get("codebert_similarity", 0.0)),
                    cross_encoder_score=float(c.get("cross_encoder_score", 0.0)),
                    hybrid_score=float(c.get("hybrid_score", 0.0)),
                    metadata=c.get("metadata", {}),
                )
            )

        # If no contexts found, surface a controlled response
        if not search_results:
            generated = {
                "generated_code": "",
                "validation_status": "NO_CONTEXT",
                "confidence_score": 0.0,
                "generation_metadata": {
                    "error": "No retrieval context found",
                },
            }
            return GenerateResponse(
                ok=False,
                query=req.query,
                output_type=req.output_type,
                retrieval_metadata=retrieval_meta,
                retrieval_context=contexts_serialized,
                generated=generated,
            )

        # Initialize generator and produce output
        generator = SAPiFlowGenerator()
        gen_req = GenerationRequest(
            query=req.query,
            search_results=search_results,
            output_type=req.output_type,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        output = generator.generate_code(gen_req)

        generated_payload = {
            "query": output.query,
            "output_type": output.output_type,
            "generated_code": output.generated_code,
            "validation_status": output.validation_status,
            "confidence_score": output.confidence_score,
            "context_chunks_used": output.context_chunks_used,
            "generation_metadata": output.generation_metadata,
        }

        return GenerateResponse(
            ok=output.validation_status != "ERROR",
            query=req.query,
            output_type=req.output_type,
            retrieval_metadata=retrieval_meta,
            retrieval_context=contexts_serialized,
            generated=generated_payload,
        )
    except HTTPException:
        raise
    except Exception as e:
        detail = {
            "message": str(e),
            "trace": traceback.format_exc().splitlines()[-5:],
        }
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "query": req.query,
                "output_type": req.output_type,
                "retrieval_metadata": {"error": detail},
                "retrieval_context": [],
                "generated": {
                    "generated_code": "",
                    "validation_status": "ERROR",
                    "confidence_score": 0.0,
                    "generation_metadata": {"error": detail},
                },
            },
        )


# Entrypoint hint for uvicorn
# uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload


