from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from utils.logging_utils import get_logger, RetrievalLogger
from utils.scoring_utils import (
    normalize_scores, 
    combine_scores, 
    calculate_hybrid_score,
    apply_boosting,
    calculate_diversity_penalty
)
from retrieval.rerankers.cross_encoder_reranker import CrossEncoderReranker
from retrieval.rerankers.metadata_reranker import MetadataReranker
from retrieval.rerankers.query_classifier import QueryClassifier


class HybridReranker:
    """
    Hybrid reranker that combines multiple ranking signals for optimal results.
    """
    
    def __init__(
        self,
        cross_encoder_reranker: Optional[CrossEncoderReranker] = None,
        metadata_reranker: Optional[MetadataReranker] = None,
        query_classifier: Optional[QueryClassifier] = None,
        weights: Optional[Dict[str, float]] = None,
        enable_diversity_penalty: bool = True,
        enable_boosting: bool = True,
        normalize_scores_flag: bool = True
    ):
        """
        Initialize the hybrid reranker.
        
        Args:
            cross_encoder_reranker: Cross-encoder reranker instance
            metadata_reranker: Metadata reranker instance
            query_classifier: Query classifier instance
            weights: Weights for different scoring components
            enable_diversity_penalty: Whether to apply diversity penalty
            enable_boosting: Whether to apply boosting
            normalize_scores_flag: Whether to normalize scores
        """
        self.logger = get_logger("hybrid_reranker")
        self.cross_encoder_reranker = cross_encoder_reranker
        self.metadata_reranker = metadata_reranker
        self.query_classifier = query_classifier
        self.enable_diversity_penalty = enable_diversity_penalty
        self.enable_boosting = enable_boosting
        self.normalize_scores_flag = normalize_scores_flag
        
        # Default weights for different components
        self.default_weights = {
            'vector': 0.25,
            'cross_encoder': 0.35,
            'metadata': 0.25,
            'query_classification': 0.15
        }
        
        # Use provided weights or defaults
        self.weights = weights or self.default_weights.copy()
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        self.logger.info("Hybrid reranker initialized")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        vector_scores: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using hybrid approach combining multiple signals.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            vector_scores: Optional pre-computed vector similarity scores
            top_k: Number of top results to return
            threshold: Minimum score threshold
            
        Returns:
            List of reranked documents with hybrid scores
        """
        with RetrievalLogger(f"Hybrid reranking for {len(documents)} documents", self.logger):
            if not documents:
                return []
            
            # Step 1: Get query classification
            query_classification = None
            if self.query_classifier:
                query_classification = self.query_classifier.classify_query(query)
            
            # Step 2: Get cross-encoder scores
            cross_encoder_scores = None
            if self.cross_encoder_reranker:
                cross_encoder_results = self.cross_encoder_reranker.rerank(
                    query, documents, normalize_scores_flag=self.normalize_scores_flag
                )
                cross_encoder_scores = [result['cross_encoder_score'] for result in cross_encoder_results]
            
            # Step 3: Get metadata scores
            metadata_scores = None
            if self.metadata_reranker:
                metadata_results = self.metadata_reranker.rerank(
                    query, documents
                )
                metadata_scores = [result['metadata_score'] for result in metadata_results]
            
            # Step 4: Calculate query classification boost
            classification_boosts = None
            if query_classification and self.query_classifier:
                classification_boosts = self._calculate_classification_boosts(
                    query_classification, documents
                )
            
            # Step 5: Combine all scores
            hybrid_scores = self._combine_all_scores(
                query, documents, vector_scores, cross_encoder_scores, 
                metadata_scores, classification_boosts
            )
            
            # Step 6: Apply diversity penalty if enabled
            if self.enable_diversity_penalty and len(documents) > 1:
                similarities = self._calculate_document_similarities(documents)
                hybrid_scores = calculate_diversity_penalty(hybrid_scores, similarities)
            
            # Step 7: Apply boosting if enabled
            if self.enable_boosting and classification_boosts:
                hybrid_scores = apply_boosting(hybrid_scores, classification_boosts)
            
            # Step 8: Create results
            results = []
            for i, (doc, score) in enumerate(zip(documents, hybrid_scores)):
                if score >= threshold:
                    result = {
                        'document': doc,
                        'hybrid_score': float(score),
                        'score': float(score),
                        'rank': i + 1
                    }
                    
                    # Add component scores if available
                    if vector_scores:
                        result['vector_score'] = vector_scores[i]
                    if cross_encoder_scores:
                        result['cross_encoder_score'] = cross_encoder_scores[i]
                    if metadata_scores:
                        result['metadata_score'] = metadata_scores[i]
                    if classification_boosts:
                        result['classification_boost'] = classification_boosts[i]
                    
                    results.append(result)
            
            # Sort by hybrid score (descending)
            results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            
            # Apply top_k limit
            if top_k is not None:
                results = results[:top_k]
            
            # Update ranks
            for i, result in enumerate(results):
                result['rank'] = i + 1
            
            self.logger.info(f"Hybrid reranked {len(results)} documents above threshold {threshold}")
            return results
    
    def _combine_all_scores(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        vector_scores: Optional[List[float]],
        cross_encoder_scores: Optional[List[float]],
        metadata_scores: Optional[List[float]],
        classification_boosts: Optional[List[float]]
    ) -> List[float]:
        """
        Combine all available scores into hybrid scores.
        
        Args:
            query: Search query
            documents: List of documents
            vector_scores: Vector similarity scores
            cross_encoder_scores: Cross-encoder scores
            metadata_scores: Metadata scores
            classification_boosts: Classification boost factors
            
        Returns:
            List of combined hybrid scores
        """
        score_components = []
        component_weights = []
        
        # Add vector scores
        if vector_scores:
            score_components.append(vector_scores)
            component_weights.append(self.weights['vector'])
        
        # Add cross-encoder scores
        if cross_encoder_scores:
            score_components.append(cross_encoder_scores)
            component_weights.append(self.weights['cross_encoder'])
        
        # Add metadata scores
        if metadata_scores:
            score_components.append(metadata_scores)
            component_weights.append(self.weights['metadata'])
        
        # Add classification scores (if no classification boosts, use neutral scores)
        if classification_boosts:
            score_components.append(classification_boosts)
            component_weights.append(self.weights['query_classification'])
        
        # Combine scores
        if score_components:
            hybrid_scores = combine_scores(
                score_components, 
                weights=component_weights,
                method='weighted_sum'
            )
        else:
            # Fallback: use uniform scores
            hybrid_scores = [1.0] * len(documents)
        
        return hybrid_scores
    
    def _calculate_classification_boosts(
        self,
        query_classification: Dict[str, Any],
        documents: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Calculate boost factors based on query classification.
        
        Args:
            query_classification: Query classification results
            documents: List of documents
            
        Returns:
            List of boost factors
        """
        primary_type = query_classification['primary_type']
        primary_confidence = query_classification.get('confidence', 0.5)
        
        boosts = []
        
        for doc in documents:
            # Extract document type from metadata
            doc_metadata = doc.get('metadata', {})
            doc_type = doc_metadata.get('component_type', 'general')
            
            # Calculate boost based on type matching
            if doc_type == primary_type:
                # Strong boost for exact type match
                boost = 0.3 * primary_confidence
            elif doc_type in query_classification.get('all_types', []):
                # Moderate boost for partial type match
                type_score = 0.5  # Default score for list-based all_types
                boost = 0.15 * type_score
            else:
                # No boost for unrelated types
                boost = 0.0
            
            boosts.append(boost)
        
        return boosts
    
    def _calculate_document_similarities(self, documents: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Calculate similarity matrix between documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Similarity matrix
        """
        # Simple text-based similarity calculation
        # In a real implementation, you might use more sophisticated methods
        
        similarities = []
        
        for i, doc1 in enumerate(documents):
            doc1_text = self._extract_document_text(doc1)
            doc1_words = set(doc1_text.lower().split())
            
            doc_similarities = []
            for j, doc2 in enumerate(documents):
                if i == j:
                    doc_similarities.append(1.0)  # Self-similarity
                else:
                    doc2_text = self._extract_document_text(doc2)
                    doc2_words = set(doc2_text.lower().split())
                    
                    # Jaccard similarity
                    intersection = len(doc1_words & doc2_words)
                    union = len(doc1_words | doc2_words)
                    
                    similarity = intersection / union if union > 0 else 0.0
                    doc_similarities.append(similarity)
            
            similarities.append(doc_similarities)
        
        return similarities
    
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
    
    def get_score_breakdown(
        self,
        query: str,
        document: Dict[str, Any],
        vector_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get detailed breakdown of scores for a single document.
        
        Args:
            query: Search query
            document: Document to analyze
            vector_score: Optional vector similarity score
            
        Returns:
            Dictionary with score breakdown
        """
        breakdown = {
            'query': query,
            'document': document,
            'component_scores': {},
            'weights': self.weights.copy(),
            'final_score': 0.0
        }
        
        # Get cross-encoder score
        if self.cross_encoder_reranker:
            cross_encoder_score = self.cross_encoder_reranker.get_document_score(query, document)
            breakdown['component_scores']['cross_encoder'] = cross_encoder_score
        
        # Get metadata score breakdown
        if self.metadata_reranker:
            metadata_breakdown = self.metadata_reranker.get_score_breakdown(query, document)
            breakdown['component_scores']['metadata'] = metadata_breakdown['combined_score']
            breakdown['metadata_breakdown'] = metadata_breakdown
        
        # Get query classification
        if self.query_classifier:
            classification = self.query_classifier.classify_query(query)
            breakdown['query_classification'] = classification
        
        # Add vector score if provided
        if vector_score is not None:
            breakdown['component_scores']['vector'] = vector_score
        
        # Calculate final score
        final_score = 0.0
        for component, score in breakdown['component_scores'].items():
            if component in self.weights:
                final_score += self.weights[component] * score
        
        breakdown['final_score'] = final_score
        
        return breakdown
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update the weights for different scoring components.
        
        Args:
            new_weights: New weights dictionary
        """
        # Update weights
        for component, weight in new_weights.items():
            if component in self.weights:
                self.weights[component] = weight
        
        # Renormalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        self.logger.info("Updated hybrid reranker weights")
    
    def get_reranker_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the hybrid reranker.
        
        Returns:
            Dictionary with reranker statistics
        """
        stats = {
            'weights': self.weights.copy(),
            'enable_diversity_penalty': self.enable_diversity_penalty,
            'enable_boosting': self.enable_boosting,
            'normalize_scores': self.normalize_scores_flag,
            'components': {}
        }
        
        # Add component statistics
        if self.cross_encoder_reranker:
            stats['components']['cross_encoder'] = self.cross_encoder_reranker.get_model_info()
        
        if self.metadata_reranker:
            stats['components']['metadata'] = {
                'complexity_weight': self.metadata_reranker.complexity_weight,
                'reusability_weight': self.metadata_reranker.reusability_weight,
                'freshness_weight': self.metadata_reranker.freshness_weight,
                'popularity_weight': self.metadata_reranker.popularity_weight
            }
        
        if self.query_classifier:
            stats['components']['query_classifier'] = self.query_classifier.get_classification_stats()
        
        return stats
