from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from utils.logging_utils import get_logger, RetrievalLogger
from utils.scoring_utils import combine_scores, normalize_scores
from retrieval.vector_search import VectorSearch
from retrieval.rerankers.hybrid_reranker import HybridReranker


class HybridSearch:
    """
    Hybrid search system combining vector search with other retrieval methods.
    """
    
    def __init__(
        self,
        vector_search: VectorSearch,
        hybrid_reranker: Optional[HybridReranker] = None,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
        enable_reranking: bool = True,
        max_initial_results: int = 100,
        max_final_results: int = 20
    ):
        """
        Initialize the hybrid search system.
        
        Args:
            vector_search: Vector search instance
            hybrid_reranker: Hybrid reranker instance
            vector_weight: Weight for vector search scores
            keyword_weight: Weight for keyword search scores
            enable_reranking: Whether to enable reranking
            max_initial_results: Maximum initial results before reranking
            max_final_results: Maximum final results after reranking
        """
        self.logger = get_logger("hybrid_search")
        self.vector_search = vector_search
        self.hybrid_reranker = hybrid_reranker
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.enable_reranking = enable_reranking
        self.max_initial_results = max_initial_results
        self.max_final_results = max_final_results
        
        # Normalize weights
        total_weight = vector_weight + keyword_weight
        self.vector_weight /= total_weight
        self.keyword_weight /= total_weight
        
        self.logger.info("Hybrid search system initialized")
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: float = 0.0,
        enable_reranking: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and keyword search.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            threshold: Minimum score threshold
            enable_reranking: Whether to enable reranking (overrides instance setting)
            
        Returns:
            List of search results with scores
        """
        with RetrievalLogger(f"Hybrid search for query: {query[:50]}...", self.logger):
            # Use instance setting if not overridden
            if enable_reranking is None:
                enable_reranking = self.enable_reranking
            
            # Step 1: Perform vector search
            vector_results = self.vector_search.search(
                query, 
                top_k=self.max_initial_results,
                threshold=0.0  # No threshold for initial search
            )
            
            if not vector_results:
                self.logger.warning("No vector search results found")
                return []
            
            # Step 2: Perform keyword search
            keyword_results = self._keyword_search(query, vector_results)
            
            # Step 3: Combine vector and keyword scores
            combined_results = self._combine_search_results(
                query, vector_results, keyword_results
            )
            
            # Step 4: Apply reranking if enabled
            if enable_reranking and self.hybrid_reranker:
                final_results = self._apply_reranking(query, combined_results)
            else:
                final_results = combined_results
            
            # Step 5: Apply threshold and limit
            filtered_results = [
                result for result in final_results 
                if result['score'] >= threshold
            ]
            
            # Apply top_k limit
            if top_k is not None:
                filtered_results = filtered_results[:top_k]
            elif self.max_final_results:
                filtered_results = filtered_results[:self.max_final_results]
            
            self.logger.info(f"Hybrid search returned {len(filtered_results)} results")
            return filtered_results
    
    def _keyword_search(
        self,
        query: str,
        vector_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search on vector results.
        
        Args:
            query: Search query
            vector_results: Results from vector search
            
        Returns:
            List of keyword search results
        """
        # Extract documents from vector results
        documents = [result['document'] for result in vector_results]
        
        # Calculate keyword scores
        keyword_scores = []
        query_terms = set(query.lower().split())
        
        for doc in documents:
            # Extract document text
            doc_text = self._extract_document_text(doc)
            doc_terms = set(doc_text.lower().split())
            
            # Calculate term overlap
            overlap = len(query_terms & doc_terms)
            total_query_terms = len(query_terms)
            
            # Calculate keyword score
            if total_query_terms > 0:
                keyword_score = overlap / total_query_terms
            else:
                keyword_score = 0.0
            
            keyword_scores.append(keyword_score)
        
        # Create keyword results
        keyword_results = []
        for i, (result, keyword_score) in enumerate(zip(vector_results, keyword_scores)):
            keyword_result = result.copy()
            keyword_result['keyword_score'] = keyword_score
            keyword_result['score'] = keyword_score
            keyword_results.append(keyword_result)
        
        return keyword_results
    
    def _combine_search_results(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine vector and keyword search results.
        
        Args:
            query: Search query
            vector_results: Vector search results
            keyword_results: Keyword search results
            
        Returns:
            List of combined results
        """
        combined_results = []
        
        for vector_result, keyword_result in zip(vector_results, keyword_results):
            # Extract scores
            vector_score = vector_result.get('similarity_score', 0.0)
            keyword_score = keyword_result.get('keyword_score', 0.0)
            
            # Combine scores
            combined_score = (
                self.vector_weight * vector_score +
                self.keyword_weight * keyword_score
            )
            
            # Create combined result
            combined_result = {
                'document': vector_result['document'],
                'vector_score': vector_score,
                'keyword_score': keyword_score,
                'combined_score': combined_score,
                'score': combined_score,
                'index': vector_result.get('index')
            }
            
            combined_results.append(combined_result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return combined_results
    
    def _apply_reranking(
        self,
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply hybrid reranking to search results.
        
        Args:
            query: Search query
            search_results: Initial search results
            
        Returns:
            List of reranked results
        """
        # Extract documents and vector scores
        documents = [result['document'] for result in search_results]
        vector_scores = [result['vector_score'] for result in search_results]
        
        # Apply hybrid reranking
        reranked_results = self.hybrid_reranker.rerank(
            query=query,
            documents=documents,
            vector_scores=vector_scores,
            top_k=self.max_final_results,
            threshold=0.0
        )
        
        return reranked_results
    
    def _extract_document_text(self, document: Dict[str, Any]) -> str:
        """
        Extract text content from document.
        
        Args:
            document: Document dictionary
            
        Returns:
            Extracted text content
        """
        if isinstance(document, dict):
            # Try different text fields
            text_fields = ['text', 'content', 'body', 'description', 'title']
            for field in text_fields:
                if field in document and document[field]:
                    return str(document[field])
            
            # If no text field found, try to convert to string
            return str(document)
        else:
            return str(document)
    
    def batch_search(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        threshold: float = 0.0,
        enable_reranking: Optional[bool] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform batch hybrid search for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of top results per query
            threshold: Minimum score threshold
            enable_reranking: Whether to enable reranking
            
        Returns:
            List of result lists for each query
        """
        with RetrievalLogger(f"Batch hybrid search for {len(queries)} queries", self.logger):
            all_results = []
            
            for query in queries:
                query_results = self.search(
                    query=query,
                    top_k=top_k,
                    threshold=threshold,
                    enable_reranking=enable_reranking
                )
                all_results.append(query_results)
            
            return all_results
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the hybrid search system.
        
        Returns:
            Dictionary with search statistics
        """
        stats = {
            'vector_weight': self.vector_weight,
            'keyword_weight': self.keyword_weight,
            'enable_reranking': self.enable_reranking,
            'max_initial_results': self.max_initial_results,
            'max_final_results': self.max_final_results,
            'vector_search_stats': self.vector_search.get_index_stats()
        }
        
        if self.hybrid_reranker:
            stats['hybrid_reranker_stats'] = self.hybrid_reranker.get_reranker_stats()
        
        return stats
    
    def update_weights(self, vector_weight: float, keyword_weight: float) -> None:
        """
        Update the weights for vector and keyword search.
        
        Args:
            vector_weight: New vector search weight
            keyword_weight: New keyword search weight
        """
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        
        # Normalize weights
        total_weight = vector_weight + keyword_weight
        self.vector_weight /= total_weight
        self.keyword_weight /= total_weight
        
        self.logger.info("Updated hybrid search weights")
    
    def set_reranking_enabled(self, enabled: bool) -> None:
        """
        Enable or disable reranking.
        
        Args:
            enabled: Whether to enable reranking
        """
        self.enable_reranking = enabled
        self.logger.info(f"Reranking {'enabled' if enabled else 'disabled'}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the search index.
        
        Args:
            documents: List of documents to add
        """
        self.vector_search.add_documents(documents)
        self.logger.info(f"Added {len(documents)} documents to hybrid search index")
    
    def clear_index(self) -> None:
        """Clear the search index."""
        self.vector_search.clear_index()
        self.logger.info("Cleared hybrid search index")
    
    def save_index(self) -> None:
        """Save the search index."""
        self.vector_search.save_index()
        self.logger.info("Saved hybrid search index")
    
    def load_index(self) -> None:
        """Load the search index."""
        self.vector_search.load_index()
        self.logger.info("Loaded hybrid search index")
