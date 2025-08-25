import torch
from typing import List, Dict, Any, Optional, Tuple
import openai
import numpy as np

from utils.logging_utils import get_logger, RetrievalLogger
from utils.scoring_utils import normalize_scores

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker using OpenAI for improved ranking.
    """
    
    def __init__(self, openai_client: Optional[openai.OpenAI] = None, model_name: str = 'gpt-3.5-turbo', max_length: int = 512, device: Optional[str] = None, batch_size: int = 32):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            openai_client: OpenAI client instance
            model_name: OpenAI model name for reranking
            max_length: Maximum sequence length
            device: Device for computation (not used with OpenAI)
            batch_size: Batch size for processing
        """
        self.client = openai_client or openai.OpenAI()
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        logger.info(f"Cross-encoder reranker initialized with model: {model_name}")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: Optional[int] = None, threshold: float = 0.0, normalize_scores_flag: bool = True) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder model.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top results to return
            threshold: Minimum score threshold
            normalize_scores_flag: Whether to normalize scores
            
        Returns:
            Reranked documents with scores
        """
        if not documents:
            return []
        
        with RetrievalLogger(f"Cross-encoder reranking for {len(documents)} documents", logger):
            # Prepare query-document pairs
            pairs = self._prepare_pairs(query, documents)
            
            # Get scores from cross-encoder
            scores = self._get_scores(pairs)
            
            # Normalize scores if requested
            if normalize_scores_flag:
                scores = normalize_scores(scores, method='minmax')
            
            # Combine documents with scores
            results = []
            for doc, score in zip(documents, scores):
                if score >= threshold:
                    result = doc.copy()
                    result['cross_encoder_score'] = score
                    results.append(result)
            
            # Sort by score (descending)
            results.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
            
            # Apply top_k limit
            if top_k is not None:
                results = results[:top_k]
            
            logger.info(f"Reranked {len(results)} documents above threshold {threshold}")
            return results
    
    def batch_rerank(self, queries: List[str], documents_list: List[List[Dict[str, Any]]], top_k: Optional[int] = None, threshold: float = 0.0) -> List[List[Dict[str, Any]]]:
        """
        Perform batch reranking for multiple queries.
        
        Args:
            queries: List of search queries
            documents_list: List of document lists for each query
            top_k: Number of top results per query
            threshold: Minimum score threshold
            
        Returns:
            List of reranked document lists for each query
        """
        with RetrievalLogger(f"Batch cross-encoder reranking for {len(queries)} queries", logger):
            results = []
            for query, documents in zip(queries, documents_list):
                query_results = self.rerank(query, documents, top_k, threshold)
                results.append(query_results)
            return results
    
    def _prepare_pairs(self, query: str, documents: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """
        Prepare query-document pairs for cross-encoder input.
        
        Args:
            query: Search query
            documents: List of documents
            
        Returns:
            List of (query, document_text) pairs
        """
        pairs = []
        for doc in documents:
            # Extract document text
            doc_text = self._extract_document_text(doc)
            pairs.append((query, doc_text))
        return pairs
    
    def _extract_document_text(self, document: Dict[str, Any]) -> str:
        """
        Extract text content from document.
        
        Args:
            document: Document dictionary
            
        Returns:
            Document text
        """
        # Try different possible text fields
        text_fields = ['text', 'content', 'body', 'description', 'title']
        for field in text_fields:
            if field in document and document[field]:
                return str(document[field])
        
        # If no text field found, convert document to string
        return str(document)
    
    def _get_scores(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Get scores from OpenAI cross-encoder for query-document pairs.
        
        Args:
            pairs: List of (query, document_text) pairs
            
        Returns:
            List of relevance scores
        """
        scores = []
        
        # Process in batches
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            batch_scores = self._get_batch_scores(batch_pairs)
            scores.extend(batch_scores)
        
        return scores
    
    def _get_batch_scores(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Get scores for a batch of query-document pairs.
        
        Args:
            pairs: List of (query, document_text) pairs
            
        Returns:
            List of relevance scores
        """
        scores = []
        
        for query, doc_text in pairs:
            try:
                # Truncate document text if too long
                if len(doc_text) > self.max_length:
                    doc_text = doc_text[:self.max_length] + "..."
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a document relevance scorer. Rate how relevant a document is to a query on a scale from 0 to 1, where 0 means completely irrelevant and 1 means perfectly relevant. Consider semantic meaning, context, and information overlap. Return only the numeric score."},
                        {"role": "user", "content": f"Rate the relevance of this document to the query:\nQuery: {query}\nDocument: {doc_text}"}
                    ],
                    max_tokens=10,
                    temperature=0.1
                )
                
                score_text = response.choices[0].message.content.strip()
                score = float(score_text)
                score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
                scores.append(score)
                
            except Exception as e:
                logger.warning(f"Failed to get score for pair: {e}")
                scores.append(0.0)  # Default score
        
        return scores
    
    def get_document_score(self, query: str, document: Dict[str, Any]) -> float:
        """
        Get cross-encoder score for a single query-document pair.
        
        Args:
            query: Search query
            document: Document dictionary
            
        Returns:
            Cross-encoder score
        """
        doc_text = self._extract_document_text(document)
        scores = self._get_scores([(query, doc_text)])
        return scores[0] if scores else 0.0
    
    def compute_similarity_matrix(self, documents: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Compute similarity matrix between documents using cross-encoder.
        
        Args:
            documents: List of documents
            
        Returns:
            Similarity matrix
        """
        n = len(documents)
        similarity_matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                # Use document text as query for comparison
                doc_i_text = self._extract_document_text(documents[i])
                doc_j_text = self._extract_document_text(documents[j])
                
                # Get similarity score
                similarity = self._get_scores([(doc_i_text, doc_j_text)])[0]
                
                # Symmetric matrix
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
            
            # Diagonal is 1.0 (self-similarity)
            similarity_matrix[i][i] = 1.0
        
        return similarity_matrix
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the cross-encoder model.
        
        Returns:
            Model information dictionary
        """
        return {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'model_type': 'openai_cross_encoder'
        }
    
    def update_batch_size(self, new_batch_size: int) -> None:
        """
        Update the batch size for processing.
        
        Args:
            new_batch_size: New batch size
        """
        self.batch_size = new_batch_size
        logger.info(f"Updated batch size to {new_batch_size}")
    
    def to_device(self, device: str) -> None:
        """
        Move model to specified device (not applicable for OpenAI).
        
        Args:
            device: Target device
        """
        logger.info(f"Cross-encoder model is cloud-based, device setting not applicable")


class CrossEncoderRerankerWithCache(CrossEncoderReranker):
    """
    Cross-encoder reranker with caching for improved performance.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
        self.cache_size = 1000
        logger.info("Cross-encoder reranker with caching initialized")
    
    def _get_cached_score(self, query: str, document_text: str) -> Optional[float]:
        """
        Get cached score for query-document pair.
        
        Args:
            query: Search query
            document_text: Document text
            
        Returns:
            Cached score or None if not found
        """
        cache_key = f"{query}|||{document_text[:100]}"  # Truncate for key
        return self.cache.get(cache_key)
    
    def _cache_score(self, query: str, document_text: str, score: float) -> None:
        """
        Cache score for query-document pair.
        
        Args:
            query: Search query
            document_text: Document text
            score: Relevance score
        """
        cache_key = f"{query}|||{document_text[:100]}"  # Truncate for key
        
        # Simple LRU cache implementation
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple approach)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = score
    
    def _get_scores(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Get scores with caching support.
        
        Args:
            pairs: List of (query, document_text) pairs
            
        Returns:
            List of relevance scores
        """
        scores = []
        
        for query, doc_text in pairs:
            # Check cache first
            cached_score = self._get_cached_score(query, doc_text)
            if cached_score is not None:
                scores.append(cached_score)
                continue
            
            # Get score from OpenAI
            try:
                if len(doc_text) > self.max_length:
                    doc_text = doc_text[:self.max_length] + "..."
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a document relevance scorer. Rate how relevant a document is to a query on a scale from 0 to 1, where 0 means completely irrelevant and 1 means perfectly relevant. Consider semantic meaning, context, and information overlap. Return only the numeric score."},
                        {"role": "user", "content": f"Rate the relevance of this document to the query:\nQuery: {query}\nDocument: {doc_text}"}
                    ],
                    max_tokens=10,
                    temperature=0.1
                )
                
                score_text = response.choices[0].message.content.strip()
                score = float(score_text)
                score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
                
                # Cache the score
                self._cache_score(query, doc_text, score)
                scores.append(score)
                
            except Exception as e:
                logger.warning(f"Failed to get score for pair: {e}")
                scores.append(0.0)
        
        return scores
    
    def clear_cache(self) -> None:
        """Clear the score cache."""
        self.cache.clear()
        logger.info("Cleared cross-encoder score cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size,
            'cache_hit_rate': 0.0  # Would need to track hits/misses for accurate rate
        }
