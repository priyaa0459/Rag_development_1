import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def normalize_scores(scores: List[float], method: str = 'minmax') -> List[float]:
    """
    Normalize scores using various methods.
    
    Args:
        scores: List of scores to normalize
        method: Normalization method ('minmax', 'standard', 'rank', 'softmax')
        
    Returns:
        List of normalized scores
    """
    if not scores:
        return []
    
    scores = np.array(scores)
    
    if method == 'minmax':
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return ((scores - min_score) / (max_score - min_score)).tolist()
    
    elif method == 'standard':
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        if std_score == 0:
            return [0.0] * len(scores)
        return ((scores - mean_score) / std_score).tolist()
    
    elif method == 'rank':
        # Convert to rank-based scores (1 = best, n = worst)
        ranks = np.argsort(np.argsort(-scores)) + 1
        return (1.0 / ranks).tolist()
    
    elif method == 'softmax':
        # Softmax normalization
        exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
        return (exp_scores / np.sum(exp_scores)).tolist()
    
    else:
        logger.warning(f"Unknown normalization method: {method}. Using minmax.")
        return normalize_scores(scores, 'minmax')


def combine_scores(score_lists: List[List[float]], weights: Optional[List[float]] = None, method: str = 'weighted_sum') -> List[float]:
    """
    Combine multiple score lists using various methods.
    
    Args:
        score_lists: List of score lists to combine
        weights: Optional weights for each score list
        method: Combination method ('weighted_sum', 'product', 'max', 'min')
        
    Returns:
        Combined scores
    """
    if not score_lists:
        return []
    
    # Ensure all score lists have the same length
    lengths = [len(scores) for scores in score_lists]
    if len(set(lengths)) > 1:
        logger.warning("Score lists have different lengths. Padding with zeros.")
        max_length = max(lengths)
        score_lists = [scores + [0.0] * (max_length - len(scores)) for scores in score_lists]
    
    # Default weights
    if weights is None:
        weights = [1.0 / len(score_lists)] * len(score_lists)
    
    # Normalize weights
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    
    # Convert to numpy arrays
    score_arrays = [np.array(scores) for scores in score_lists]
    
    if method == 'weighted_sum':
        combined = np.zeros_like(score_arrays[0])
        for scores, weight in zip(score_arrays, weights):
            combined += scores * weight
        return combined.tolist()
    
    elif method == 'product':
        combined = np.ones_like(score_arrays[0])
        for scores, weight in zip(score_arrays, weights):
            combined *= (scores * weight + (1 - weight))  # Weighted product
        return combined.tolist()
    
    elif method == 'max':
        combined = np.maximum.reduce(score_arrays)
        return combined.tolist()
    
    elif method == 'min':
        combined = np.minimum.reduce(score_arrays)
        return combined.tolist()
    
    else:
        logger.warning(f"Unknown combination method: {method}. Using weighted_sum.")
        return combine_scores(score_lists, weights, 'weighted_sum')


def calculate_rerank_score(original_score: float, rerank_score: float, alpha: float = 0.7) -> float:
    """
    Calculate combined rerank score from original and rerank scores.
    
    Args:
        original_score: Original ranking score
        rerank_score: Reranking score
        alpha: Weight for rerank score (0-1)
        
    Returns:
        Combined rerank score
    """
    return alpha * rerank_score + (1 - alpha) * original_score


def apply_boosting(scores: List[float], boost_factors: List[float], boost_threshold: float = 0.5) -> List[float]:
    """
    Apply boosting to scores based on boost factors.
    
    Args:
        scores: Original scores
        boost_factors: Boost factors for each score
        boost_threshold: Threshold for applying boost
        
    Returns:
        Boosted scores
    """
    if len(scores) != len(boost_factors):
        logger.warning("Scores and boost factors have different lengths")
        return scores
    
    boosted_scores = []
    for score, boost_factor in zip(scores, boost_factors):
        if boost_factor > boost_threshold:
            # Apply boost: increase score based on boost factor
            boost_amount = boost_factor * 0.3  # Max 30% boost
            boosted_score = score * (1 + boost_amount)
            boosted_scores.append(min(boosted_score, 1.0))  # Cap at 1.0
        else:
            boosted_scores.append(score)
    
    return boosted_scores


def calculate_diversity_penalty(scores: List[float], similarities: List[List[float]], penalty_weight: float = 0.1) -> List[float]:
    """
    Calculate diversity penalty to reduce redundancy in results.
    
    Args:
        scores: Original scores
        similarities: Similarity matrix between documents
        penalty_weight: Weight for diversity penalty
        
    Returns:
        Scores with diversity penalty applied
    """
    if not similarities or len(scores) != len(similarities):
        return scores
    
    penalized_scores = []
    for i, score in enumerate(scores):
        # Calculate penalty based on similarity to higher-ranked documents
        penalty = 0.0
        for j in range(i):
            if j < len(similarities[i]):
                similarity = similarities[i][j]
                # Higher penalty for more similar documents
                penalty += similarity * penalty_weight
        
        # Apply penalty
        penalized_score = score * (1 - penalty)
        penalized_scores.append(max(penalized_score, 0.0))
    
    return penalized_scores


def calculate_hybrid_score(vector_score: float, keyword_score: float, metadata_score: float, cross_encoder_score: float, weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate hybrid score combining multiple scoring signals.
    
    Args:
        vector_score: Vector similarity score
        keyword_score: Keyword matching score
        metadata_score: Metadata-based score
        cross_encoder_score: Cross-encoder score
        weights: Optional weights for each score type
        
    Returns:
        Hybrid score
    """
    if weights is None:
        weights = {
            'vector': 0.4,
            'keyword': 0.2,
            'metadata': 0.2,
            'cross_encoder': 0.2
        }
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    hybrid_score = (
        vector_score * weights['vector'] +
        keyword_score * weights['keyword'] +
        metadata_score * weights['metadata'] +
        cross_encoder_score * weights['cross_encoder']
    )
    
    return max(0.0, min(1.0, hybrid_score))


def calculate_confidence_score(scores: List[float], similarities: List[List[float]]) -> List[float]:
    """
    Calculate confidence scores based on score consistency and diversity.
    
    Args:
        scores: Document scores
        similarities: Similarity matrix
        
    Returns:
        Confidence scores
    """
    if not scores:
        return []
    
    confidence_scores = []
    for i, score in enumerate(scores):
        # Base confidence on score value
        base_confidence = score
        
        # Adjust based on similarity to other documents
        if i < len(similarities):
            avg_similarity = np.mean(similarities[i]) if similarities[i] else 0.0
            # Higher confidence for more unique documents
            uniqueness_factor = 1.0 - avg_similarity
            confidence = base_confidence * (0.7 + 0.3 * uniqueness_factor)
        else:
            confidence = base_confidence
        
        confidence_scores.append(max(0.0, min(1.0, confidence)))
    
    return confidence_scores


def rank_documents(documents: List[Dict[str, Any]], scores: List[float], max_results: Optional[int] = None) -> List[Tuple[Dict[str, Any], float]]:
    """
    Rank documents by their scores.
    
    Args:
        documents: List of documents
        scores: Corresponding scores
        max_results: Maximum number of results to return
        
    Returns:
        List of (document, score) tuples ranked by score
    """
    if len(documents) != len(scores):
        logger.warning("Documents and scores have different lengths")
        return []
    
    # Create (document, score) pairs
    doc_score_pairs = list(zip(documents, scores))
    
    # Sort by score (descending)
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Apply max_results limit
    if max_results is not None:
        doc_score_pairs = doc_score_pairs[:max_results]
    
    return doc_score_pairs


def calculate_openai_similarity(text1: str, text2: str, model: str = "local") -> float:
    """
    Local semantic similarity proxy (no external APIs). Uses token overlap Jaccard.
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)


def calculate_openai_relevance_score(query: str, document_text: str, model: str = "local") -> float:
    """Local relevance proxy via keyword overlap fraction."""
    query_words = set(query.lower().split())
    doc_words = set(document_text.lower().split())
    if not query_words:
        return 0.0
    matches = sum(1 for word in query_words if word in doc_words)
    return matches / len(query_words)
