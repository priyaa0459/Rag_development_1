import re
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np

from utils.logging_utils import get_logger, RetrievalLogger
from utils.scoring_utils import normalize_scores
from utils.query_utils import calculate_query_complexity


class MetadataReranker:
    """
    Metadata-based reranker using document complexity, reusability, and other metadata features.
    """
    
    def __init__(
        self,
        complexity_weight: float = 0.3,
        reusability_weight: float = 0.3,
        freshness_weight: float = 0.2,
        popularity_weight: float = 0.2,
        normalize_scores_flag: bool = True
    ):
        """
        Initialize the metadata reranker.
        
        Args:
            complexity_weight: Weight for complexity score
            reusability_weight: Weight for reusability score
            freshness_weight: Weight for freshness score
            popularity_weight: Weight for popularity score
            normalize_scores_flag: Whether to normalize scores
        """
        self.logger = get_logger("metadata_reranker")
        self.complexity_weight = complexity_weight
        self.reusability_weight = reusability_weight
        self.freshness_weight = freshness_weight
        self.popularity_weight = popularity_weight
        self.normalize_scores_flag = normalize_scores_flag
        
        # Normalize weights
        total_weight = complexity_weight + reusability_weight + freshness_weight + popularity_weight
        self.complexity_weight /= total_weight
        self.reusability_weight /= total_weight
        self.freshness_weight /= total_weight
        self.popularity_weight /= total_weight
        
        self.logger.info("Metadata reranker initialized")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on metadata features.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top results to return
            threshold: Minimum score threshold
            
        Returns:
            List of reranked documents with metadata scores
        """
        with RetrievalLogger(f"Metadata reranking for {len(documents)} documents", self.logger):
            if not documents:
                return []
            
            # Calculate metadata scores for each document
            metadata_scores = []
            for doc in documents:
                scores = self._calculate_document_scores(query, doc)
                metadata_scores.append(scores)
            
            # Combine scores
            combined_scores = self._combine_metadata_scores(metadata_scores)
            
            # Normalize if requested
            if self.normalize_scores_flag:
                combined_scores = normalize_scores(combined_scores, method='minmax')
            
            # Create results
            results = []
            for doc, score, metadata_score in zip(documents, combined_scores, metadata_scores):
                if score >= threshold:
                    result = {
                        'document': doc,
                        'metadata_score': float(score),
                        'score': float(score),
                        'complexity_score': metadata_score['complexity'],
                        'reusability_score': metadata_score['reusability'],
                        'freshness_score': metadata_score['freshness'],
                        'popularity_score': metadata_score['popularity']
                    }
                    results.append(result)
            
            # Sort by score (descending)
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Apply top_k limit
            if top_k is not None:
                results = results[:top_k]
            
            self.logger.info(f"Metadata reranked {len(results)} documents above threshold {threshold}")
            return results
    
    def _calculate_document_scores(self, query: str, document: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate individual metadata scores for a document.
        
        Args:
            query: Search query
            document: Document to score
            
        Returns:
            Dictionary of metadata scores
        """
        scores = {}
        
        # Complexity score
        scores['complexity'] = self._calculate_complexity_score(query, document)
        
        # Reusability score
        scores['reusability'] = self._calculate_reusability_score(document)
        
        # Freshness score
        scores['freshness'] = self._calculate_freshness_score(document)
        
        # Popularity score
        scores['popularity'] = self._calculate_popularity_score(document)
        
        return scores
    
    def _calculate_complexity_score(self, query: str, document: Dict[str, Any]) -> float:
        """
        Calculate complexity score based on query-document complexity matching.
        
        Args:
            query: Search query
            document: Document to score
            
        Returns:
            Complexity score between 0 and 1
        """
        # Get query complexity
        query_complexity = calculate_query_complexity(query)
        
        # Get document complexity
        doc_text = self._extract_document_text(document)
        doc_complexity = self._calculate_document_complexity(doc_text)
        
        # Calculate complexity match score
        # Higher score when query and document complexity are similar
        complexity_diff = abs(query_complexity - doc_complexity)
        complexity_score = 1.0 - complexity_diff
        
        return max(0.0, complexity_score)
    
    def _calculate_document_complexity(self, text: str) -> float:
        """
        Calculate complexity score for document text.
        
        Args:
            text: Document text
            
        Returns:
            Complexity score between 0 and 1
        """
        if not text:
            return 0.0
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Calculate complexity factors
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        unique_words = len(set(text.lower().split()))
        total_words = len(text.split())
        lexical_diversity = unique_words / max(total_words, 1)
        
        # Technical terms (simple heuristic)
        technical_terms = len(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', text))
        technical_ratio = technical_terms / max(total_words, 1)
        
        # Code blocks or technical content
        code_indicators = len(re.findall(r'[{}()\[\]]|def |class |import |function', text))
        code_ratio = code_indicators / max(total_words, 1)
        
        # Combine factors
        complexity_factors = [
            min(avg_sentence_length / 20, 1.0),  # Normalize by expected max
            lexical_diversity,
            min(technical_ratio * 10, 1.0),  # Scale up technical terms
            min(code_ratio * 5, 1.0)  # Scale up code indicators
        ]
        
        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]
        complexity = sum(factor * weight for factor, weight in zip(complexity_factors, weights))
        
        return min(complexity, 1.0)
    
    def _calculate_reusability_score(self, document: Dict[str, Any]) -> float:
        """
        Calculate reusability score based on document characteristics.
        
        Args:
            document: Document to score
            
        Returns:
            Reusability score between 0 and 1
        """
        doc_text = self._extract_document_text(document)
        
        # Factors that indicate high reusability
        reusability_factors = []
        
        # 1. Template-like structure
        template_indicators = len(re.findall(r'\{.*?\}|\[.*?\]|___+', doc_text))
        template_score = min(template_indicators / 10, 1.0)
        reusability_factors.append(template_score)
        
        # 2. Step-by-step instructions
        step_indicators = len(re.findall(r'\b(step|step \d+|first|second|third|finally|next|then)\b', doc_text.lower()))
        step_score = min(step_indicators / 5, 1.0)
        reusability_factors.append(step_score)
        
        # 3. Configuration examples
        config_indicators = len(re.findall(r'config|setting|parameter|option|example', doc_text.lower()))
        config_score = min(config_indicators / 3, 1.0)
        reusability_factors.append(config_score)
        
        # 4. Code examples
        code_indicators = len(re.findall(r'```|`.*?`|def |class |function', doc_text))
        code_score = min(code_indicators / 5, 1.0)
        reusability_factors.append(code_score)
        
        # 5. Generic language (not specific to one use case)
        specific_terms = len(re.findall(r'\b(specific|particular|this case|only|unique)\b', doc_text.lower()))
        generic_score = max(0.0, 1.0 - specific_terms / 5)
        reusability_factors.append(generic_score)
        
        # Weighted average
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        reusability = sum(factor * weight for factor, weight in zip(reusability_factors, weights))
        
        return min(reusability, 1.0)
    
    def _calculate_freshness_score(self, document: Dict[str, Any]) -> float:
        """
        Calculate freshness score based on document age and update frequency.
        
        Args:
            document: Document to score
            
        Returns:
            Freshness score between 0 and 1
        """
        # Extract date information from metadata
        metadata = document.get('metadata', {})
        
        # Check for various date fields
        date_fields = ['created_date', 'updated_date', 'last_modified', 'date', 'timestamp']
        dates = []
        
        for field in date_fields:
            if field in metadata:
                try:
                    # Simple date parsing (you might want to use dateutil for more robust parsing)
                    date_str = str(metadata[field])
                    if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                        dates.append(date_str)
                except:
                    continue
        
        if not dates:
            # No date information available
            return 0.5  # Neutral score
        
        # Use the most recent date
        latest_date = max(dates)
        
        # Calculate days since latest update (simplified)
        # In a real implementation, you'd use proper date parsing
        try:
            from datetime import datetime
            latest_dt = datetime.strptime(latest_date, '%Y-%m-%d')
            current_dt = datetime.now()
            days_diff = (current_dt - latest_dt).days
            
            # Convert to freshness score (newer = higher score)
            freshness = max(0.0, 1.0 - (days_diff / 365))  # Decay over 1 year
        except:
            freshness = 0.5
        
        return freshness
    
    def _calculate_popularity_score(self, document: Dict[str, Any]) -> float:
        """
        Calculate popularity score based on usage statistics.
        
        Args:
            document: Document to score
            
        Returns:
            Popularity score between 0 and 1
        """
        metadata = document.get('metadata', {})
        
        # Extract popularity indicators
        views = metadata.get('views', 0)
        downloads = metadata.get('downloads', 0)
        likes = metadata.get('likes', 0)
        shares = metadata.get('shares', 0)
        rating = metadata.get('rating', 0)
        
        # Normalize each metric
        max_views = 10000  # Adjust based on your data
        max_downloads = 1000
        max_likes = 500
        max_shares = 100
        max_rating = 5
        
        normalized_views = min(views / max_views, 1.0)
        normalized_downloads = min(downloads / max_downloads, 1.0)
        normalized_likes = min(likes / max_likes, 1.0)
        normalized_shares = min(shares / max_shares, 1.0)
        normalized_rating = rating / max_rating
        
        # Weighted combination
        popularity_factors = [
            normalized_views * 0.3,
            normalized_downloads * 0.3,
            normalized_likes * 0.2,
            normalized_shares * 0.1,
            normalized_rating * 0.1
        ]
        
        popularity = sum(popularity_factors)
        return min(popularity, 1.0)
    
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
    
    def _combine_metadata_scores(self, metadata_scores: List[Dict[str, float]]) -> List[float]:
        """
        Combine individual metadata scores into final scores.
        
        Args:
            metadata_scores: List of metadata score dictionaries
            
        Returns:
            List of combined scores
        """
        combined_scores = []
        
        for scores in metadata_scores:
            combined_score = (
                self.complexity_weight * scores['complexity'] +
                self.reusability_weight * scores['reusability'] +
                self.freshness_weight * scores['freshness'] +
                self.popularity_weight * scores['popularity']
            )
            combined_scores.append(combined_score)
        
        return combined_scores
    
    def get_score_breakdown(self, query: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed breakdown of metadata scores for a document.
        
        Args:
            query: Search query
            document: Document to analyze
            
        Returns:
            Dictionary with score breakdown
        """
        scores = self._calculate_document_scores(query, document)
        combined_score = sum([
            self.complexity_weight * scores['complexity'],
            self.reusability_weight * scores['reusability'],
            self.freshness_weight * scores['freshness'],
            self.popularity_weight * scores['popularity']
        ])
        
        return {
            'combined_score': combined_score,
            'complexity_score': scores['complexity'],
            'reusability_score': scores['reusability'],
            'freshness_score': scores['freshness'],
            'popularity_score': scores['popularity'],
            'weights': {
                'complexity': self.complexity_weight,
                'reusability': self.reusability_weight,
                'freshness': self.freshness_weight,
                'popularity': self.popularity_weight
            }
        }
    
    def update_weights(
        self,
        complexity_weight: Optional[float] = None,
        reusability_weight: Optional[float] = None,
        freshness_weight: Optional[float] = None,
        popularity_weight: Optional[float] = None
    ) -> None:
        """
        Update the weights for different metadata factors.
        
        Args:
            complexity_weight: New complexity weight
            reusability_weight: New reusability weight
            freshness_weight: New freshness weight
            popularity_weight: New popularity weight
        """
        if complexity_weight is not None:
            self.complexity_weight = complexity_weight
        if reusability_weight is not None:
            self.reusability_weight = reusability_weight
        if freshness_weight is not None:
            self.freshness_weight = freshness_weight
        if popularity_weight is not None:
            self.popularity_weight = popularity_weight
        
        # Renormalize weights
        total_weight = (self.complexity_weight + self.reusability_weight + 
                       self.freshness_weight + self.popularity_weight)
        self.complexity_weight /= total_weight
        self.reusability_weight /= total_weight
        self.freshness_weight /= total_weight
        self.popularity_weight /= total_weight
        
        self.logger.info("Updated metadata reranker weights")
