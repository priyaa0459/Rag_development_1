"""
SAP iFlow RAG Retrieval and Re-ranking System (Person 2)
Person 2: Retrieval and Re-ranking Layer

This module implements:
- CodeBERT-based vector similarity search using Supabase
- Cross-encoder re-ranking for improved relevance
- Hybrid scoring combining semantic and re-rank scores
- Enriched result output with metadata
"""

import os
import json
import time
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import CrossEncoder
import numpy as np

# Load environment variables
load_dotenv()

@dataclass
class SearchResult:
    """Structured representation of a search result."""
    chunk_id: str
    instruction: str
    input_context: str
    output_code: str
    output_type: str
    embedding_model: str
    codebert_similarity: float
    cross_encoder_score: float
    hybrid_score: float
    metadata: Dict[str, Any]
    
    def __str__(self):
        return (f"Result: {self.chunk_id}\n"
                f"  CodeBERT Similarity: {self.codebert_similarity:.3f}\n"
                f"  Cross-Encoder Score: {self.cross_encoder_score:.3f}\n"
                f"  Hybrid Score: {self.hybrid_score:.3f}\n"
                f"  Type: {self.output_type}\n"
                f"  Instruction: {self.instruction[:100]}...\n"
                f"  Model: {self.embedding_model}")

class SAPiFlowRetriever:
    """
    Advanced retrieval system for SAP iFlow RAG pipeline.
    Combines CodeBERT vector search with cross-encoder re-ranking.
    """
    
    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize the retriever with Supabase connection."""
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.cross_encoder = None
        self.models_initialized = False
        
    def initialize_models(self):
        """Initialize the cross-encoder model for re-ranking."""
        print("Initializing cross-encoder re-ranking model...")
        try:
            # Use ms-marco-MiniLM-L-6-v2 for cross-encoder re-ranking
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.models_initialized = True
            print("✓ Cross-encoder model loaded: ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"Error loading cross-encoder model: {e}")
            print("Falling back to CodeBERT-only retrieval")
            self.models_initialized = False
    
    def retrieve_candidates(self, query: str, top_k: int = 10, 
                          match_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Retrieve candidate chunks using CodeBERT vector similarity search.
        
        Args:
            query: User query string
            top_k: Number of top results to retrieve
            match_threshold: Minimum similarity threshold (0.5 for relevance)
            
        Returns:
            List of candidate chunks with metadata
        """
        try:
            # Generate query embedding using CodeBERT (reuse from vector_loader)
            try:
                from vector_loader_cohere import SupabaseVectorLoader
            except ImportError:
                from src.vector_loader_cohere import SupabaseVectorLoader
            
            # Create temporary loader for embedding generation
            temp_loader = SupabaseVectorLoader(self.supabase.supabase_url, self.supabase.supabase_key)
            temp_loader.initialize_models()
            query_embedding = temp_loader.generate_codebert_embedding(query)
            
            print(f"Retrieving top {top_k} candidates with threshold {match_threshold}...")
            
            # Use the existing search_sap_iflow_chunks RPC function
            result = self.supabase.rpc('search_sap_iflow_chunks', {
                'query_embedding': query_embedding,
                'match_threshold': match_threshold,
                'match_count': top_k * 2,  # Get more candidates for re-ranking
                'filter_model': 'codebert-base'
            }).execute()
            
            candidates = result.data or []
            
            if not candidates:
                print(f"No candidates found above threshold {match_threshold}")
                return []
            
            print(f"Retrieved {len(candidates)} candidates for re-ranking")
            return candidates
            
        except Exception as e:
            print(f"Error in candidate retrieval: {e}")
            return []
    
    def re_rank_candidates(self, query: str, candidates: List[Dict[str, Any]], 
                          top_k: int = 5) -> List[SearchResult]:
        """
        Re-rank candidates using cross-encoder model.
        
        Args:
            query: Original user query
            candidates: List of candidate chunks from vector search
            top_k: Final number of results to return
            
        Returns:
            List of re-ranked SearchResult objects
        """
        if not self.models_initialized or not self.cross_encoder:
            print("Cross-encoder not available, returning CodeBERT-only results")
            return self._create_search_results(query, candidates, top_k)
        
        try:
            print("Applying cross-encoder re-ranking...")
            
            # Prepare query-document pairs for cross-encoder
            query_doc_pairs = []
            for candidate in candidates:
                # Combine instruction and input context for better context
                context = f"{candidate.get('instruction', '')} {candidate.get('input_context', '')}"
                query_doc_pairs.append([query, context])
            
            # Get cross-encoder scores
            cross_encoder_scores = self.cross_encoder.predict(query_doc_pairs)
            
            # Create SearchResult objects with both scores
            results = []
            for i, candidate in enumerate(candidates):
                result = SearchResult(
                    chunk_id=candidate.get('chunk_id', ''),
                    instruction=candidate.get('instruction', ''),
                    input_context=candidate.get('input_context', ''),
                    output_code=candidate.get('output_code', ''),
                    output_type=candidate.get('output_type', ''),
                    embedding_model=candidate.get('embedding_model', ''),
                    codebert_similarity=candidate.get('similarity', 0.0),
                    cross_encoder_score=float(cross_encoder_scores[i]),
                    hybrid_score=0.0,  # Will be calculated below
                    metadata=candidate.get('metadata', {})
                )
                results.append(result)
            
            # Calculate hybrid scores (weighted combination)
            results = self._calculate_hybrid_scores(results)
            
            # Sort by hybrid score and return top_k
            results.sort(key=lambda x: x.hybrid_score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            print(f"Error in cross-encoder re-ranking: {e}")
            return self._create_search_results(query, candidates, top_k)
    
    def _calculate_hybrid_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Calculate hybrid scores combining CodeBERT similarity and cross-encoder scores.
        
        Args:
            results: List of SearchResult objects
            
        Returns:
            Updated results with hybrid scores
        """
        if not results:
            return results
        
        # Normalize scores to 0-1 range
        codebert_scores = [r.codebert_similarity for r in results]
        cross_encoder_scores = [r.cross_encoder_score for r in results]
        
        # Min-max normalization
        def normalize_scores(scores):
            if not scores or max(scores) == min(scores):
                return [0.5] * len(scores)
            return [(s - min(scores)) / (max(scores) - min(scores)) for s in scores]
        
        norm_codebert = normalize_scores(codebert_scores)
        norm_cross_encoder = normalize_scores(cross_encoder_scores)
        
        # Calculate hybrid scores (weighted combination)
        # Adjust weights based on your preference
        codebert_weight = 0.4
        cross_encoder_weight = 0.6
        
        for i, result in enumerate(results):
            hybrid_score = (codebert_weight * norm_codebert[i] + 
                          cross_encoder_weight * norm_cross_encoder[i])
            result.hybrid_score = hybrid_score
        
        return results
    
    def _create_search_results(self, query: str, candidates: List[Dict[str, Any]], 
                             top_k: int) -> List[SearchResult]:
        """Create SearchResult objects without cross-encoder scoring."""
        results = []
        for candidate in candidates[:top_k]:
            result = SearchResult(
                chunk_id=candidate.get('chunk_id', ''),
                instruction=candidate.get('instruction', ''),
                input_context=candidate.get('input_context', ''),
                output_code=candidate.get('output_code', ''),
                output_type=candidate.get('output_type', ''),
                embedding_model=candidate.get('embedding_model', ''),
                codebert_similarity=candidate.get('similarity', 0.0),
                cross_encoder_score=0.0,
                hybrid_score=candidate.get('similarity', 0.0),  # Use CodeBERT score as hybrid
                metadata=candidate.get('metadata', {})
            )
            results.append(result)
        
        return results
    
    def looks_like_person_name(self, text: str) -> bool:
        """Heuristic to detect person-name-like queries (suppress retrieval)."""
        cleaned = text.strip()
        # 1-2 tokens, alphabetic, capitalized or all lowercase short name
        tokens = [t for t in re.split(r"\W+", cleaned) if t]
        if 1 <= len(tokens) <= 2:
            if all(re.fullmatch(r"[A-Za-z]{2,}", t or "") for t in tokens):
                # Avoid common domain words
                domain_words = {
                    'invoice','billing','iflow','sap','soap','rest','odata','integration','adapter',
                    'groovy','script','payload','message','endpoint','authentication','token','retry',
                    'mapping','xml','json','queue','kafka','http','oauth','tenant','subaccount','deployment'
                }
                lowered = {t.lower() for t in tokens}
                if lowered.isdisjoint(domain_words):
                    return True
        return False

    def has_domain_keyword_overlap(self, text: str) -> bool:
        """Check if query has any SAP/iFlow domain keyword overlap."""
        domain_keywords = {
            'invoice','billing','iflow','sap','soap','rest','odata','integration','adapter','groovy',
            'script','payload','message','endpoint','authentication','token','retry','mapping','xml',
            'json','queue','kafka','http','oauth','tenant','subaccount','deployment','error','log',
            'exception','credentials','keystore','certificate','sftp','proxy','hostname','port','tls'
        }
        words = {w.lower() for w in re.split(r"\W+", text) if w}
        return len(words & domain_keywords) > 0

    def search(self, query: str, top_k: int = 5, 
               match_threshold: float = 0.5) -> List[SearchResult]:
        """
        Main search method: retrieve candidates and re-rank them.
        
        Args:
            query: User query string
            top_k: Number of final results to return
            match_threshold: Minimum similarity threshold for initial retrieval
            
        Returns:
            List of re-ranked SearchResult objects
        """
        print(f"\n🔍 Searching for: '{query}'")
        print("=" * 60)
        
        # Pre-check: If the query looks like an unrelated person name and has no domain overlap, short-circuit
        if self.looks_like_person_name(query) and not self.has_domain_keyword_overlap(query):
            print("❌ No relevant results found for this query in the dataset.")
            return []
        
        # Step 1: Retrieve candidates using CodeBERT
        candidates = self.retrieve_candidates(query, top_k * 2, match_threshold)
        
        if not candidates:
            print("❌ No relevant results found for this query in the dataset.")
            return []
        
        # Step 2: Re-rank candidates using cross-encoder
        results = self.re_rank_candidates(query, candidates, top_k)
        
        # Step 3: Display results
        self._display_results(query, results)
        
        return results
    
    def _display_results(self, query: str, results: List[SearchResult]):
        """Display search results in a formatted way."""
        if not results:
            return
        
        print(f"\n📊 Found {len(results)} relevant results:")
        print("-" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.chunk_id}")
            print(f"   📈 Scores:")
            print(f"      CodeBERT: {result.codebert_similarity:.3f}")
            if result.cross_encoder_score > 0:
                print(f"      Cross-Encoder: {result.cross_encoder_score:.3f}")
            print(f"      Hybrid: {result.hybrid_score:.3f}")
            print(f"   🏷️  Type: {result.output_type}")
            print(f"   🤖 Model: {result.embedding_model}")
            print(f"   📝 Instruction: {result.instruction[:120]}...")
            
            # Show relevant metadata if available
            if result.metadata:
                meta = result.metadata
                if 'data_type' in meta:
                    print(f"   📊 Data Type: {meta['data_type']}")
                if 'source_file' in meta:
                    print(f"   📁 Source: {meta['source_file']}")
        
        print("\n" + "=" * 60)
        print("💡 These results are ready for Person 3 (LLM Generation)")
        print("🔗 Use the SearchResult objects for downstream processing")
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system."""
        try:
            # Get total chunks count
            count_result = self.supabase.table('sap_iflow_chunks').select('*', count='exact').execute()
            total_chunks = count_result.count
            
            # Get model distribution
            model_result = self.supabase.table('sap_iflow_chunks').select('embedding_model').execute()
            models = [item['embedding_model'] for item in model_result.data] if model_result.data else []
            model_distribution = {}
            for model in models:
                model_distribution[model] = model_distribution.get(model, 0) + 1
            
            stats = {
                'total_chunks': total_chunks,
                'embedding_models': model_distribution,
                'cross_encoder_available': self.models_initialized,
                'cross_encoder_model': 'ms-marco-MiniLM-L-6-v2' if self.models_initialized else 'None'
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting retrieval stats: {e}")
            return {}


def demo_search():
    """Demo function to showcase the retrieval system."""
    print("🚀 SAP iFlow RAG Retrieval System Demo")
    print("=" * 50)
    
    # Load environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Error: SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
        return
    
    # Initialize retriever
    retriever = SAPiFlowRetriever(supabase_url, supabase_key)
    retriever.initialize_models()
    
    # Get system stats
    stats = retriever.get_retrieval_stats()
    print(f"📊 System Status:")
    print(f"   Total chunks: {stats.get('total_chunks', 'Unknown')}")
    print(f"   Cross-encoder: {'✓ Available' if stats.get('cross_encoder_available') else '✗ Not available'}")
    print(f"   Model: {stats.get('cross_encoder_model', 'None')}")
    
    # Demo queries
    demo_queries = [
        "Create an invoice request integration flow",
        "SOAP parameters for iFlow configuration",
        "Groovy script for message mapping",
        "HTTP adapter authentication setup"
    ]
    
    print(f"\n🧪 Running {len(demo_queries)} demo queries...")
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*20} Demo Query {i} {'='*20}")
        results = retriever.search(query, top_k=3)
        
        if i < len(demo_queries):
            time.sleep(2)  # Brief pause between queries
    
    print(f"\n✅ Demo complete! Retrieval system is ready for Person 3 integration.")


def interactive_search():
    """Interactive search interface for testing."""
    print("🔍 Interactive SAP iFlow Search")
    print("Type 'quit' to exit, 'stats' for system info")
    
    # Load environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Error: SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
        return
    
    # Initialize retriever
    retriever = SAPiFlowRetriever(supabase_url, supabase_key)
    retriever.initialize_models()
    
    while True:
        try:
            query = input("\n🔍 Enter your query: ").strip()
            
            if query.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif query.lower() == 'stats':
                stats = retriever.get_retrieval_stats()
                print(f"📊 System Stats: {json.dumps(stats, indent=2)}")
                continue
            elif not query:
                continue
            
            # Perform search
            results = retriever.search(query, top_k=5)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_search()
        elif sys.argv[1] == "interactive":
            interactive_search()
        else:
            print("Usage: python retriever.py [demo|interactive]")
            print("  demo: Run demonstration queries")
            print("  interactive: Start interactive search interface")
    else:
        # Default: run demo
        demo_search()
