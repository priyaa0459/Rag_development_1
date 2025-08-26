import re
import string
from typing import List, Dict, Any, Optional
from collections import Counter

# Optional tiktoken import for tokenization; fallback to simple methods if unavailable
try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore

from utils.logging_utils import get_logger

logger = get_logger(__name__)

class QueryProcessor:
    """Local query processor for text preprocessing and analysis (no external APIs)."""
    
    def __init__(self, model: str = "local"):
        self.model = model
        self.encoding = None
        if tiktoken is not None and model != "local":
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except Exception:
                self.encoding = None
        
    def preprocess_query(self, query: str) -> str:
        """Preprocess query text locally."""
        return self._fallback_preprocess(query)
    
    def _fallback_preprocess(self, query: str) -> str:
        """Fallback preprocessing without OpenAI."""
        # Remove extra whitespace and convert to lowercase
        query = re.sub(r'\s+', ' ', query.strip().lower())
        # Remove special characters but keep alphanumeric and spaces
        query = re.sub(r'[^\w\s]', '', query)
        return query
    
    def tokenize_query(self, query: str) -> List[str]:
        """Tokenize query using tiktoken if available, otherwise simple split."""
        if self.encoding is not None:
            try:
                tokens = self.encoding.encode(query)
                return [self.encoding.decode([token]) for token in tokens if token != self.encoding.encode(' ')[0]]
            except Exception:
                pass
        return self._fallback_tokenize(query)
    
    def _fallback_tokenize(self, query: str) -> List[str]:
        """Fallback tokenization without OpenAI."""
        return query.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords locally."""
        return self._fallback_remove_stopwords(tokens)
    
    def _fallback_remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Fallback stopword removal without OpenAI."""
        # Basic English stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        return [token for token in tokens if token.lower() not in stopwords]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """No-op lemmatization fallback (returns original tokens)."""
        return tokens
    
    def get_query_features(self, query: str) -> Dict[str, Any]:
        """Extract query features locally."""
        return self._fallback_get_features(query)
    
    def _fallback_get_features(self, query: str) -> Dict[str, Any]:
        """Fallback feature extraction without OpenAI."""
        tokens = self.tokenize_query(query)
        words = [token for token in tokens if len(token) > 1]
        
        return {
            'tokens': tokens,
            'word_count': len(words),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'complexity_score': min(1.0, len(words) / 10.0),  # Simple heuristic
            'keywords': words[:5],  # First 5 words as keywords
            'intent': 'search',  # Default intent
            'domain': 'general'  # Default domain
        }
    
    def extract_keywords(self, query: str, top_k: int = 5) -> List[str]:
        """Extract keywords locally using frequency heuristic."""
        return self._fallback_extract_keywords(query, top_k)
    
    def _fallback_extract_keywords(self, query: str, top_k: int = 5) -> List[str]:
        """Fallback keyword extraction without OpenAI."""
        tokens = self.tokenize_query(query)
        # Simple frequency-based keyword extraction
        word_freq = Counter(tokens)
        return [word for word, _ in word_freq.most_common(top_k)]


def normalize_query(query: str) -> str:
    """Normalize query text locally (lowercase, trim, remove punctuation)."""
    processor = QueryProcessor(model="local")
    return processor.preprocess_query(query)


def extract_query_intent(query: str) -> Dict[str, Any]:
    """Heuristic intent extraction without external APIs."""
    q = query.lower()
    intent = 'search'
    if any(w in q for w in ['how to', 'guide', 'tutorial', 'steps']):
        intent = 'instruction'
    elif any(w in q for w in ['error', 'fail', 'issue']):
        intent = 'troubleshoot'
    entities = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]+', q)[:5]
    return {
        'intent': intent,
        'confidence': 0.6,
        'entities': entities,
        'action_type': 'query'
    }


def calculate_query_complexity(query: str) -> float:
    """Heuristic query complexity (0-1) based on length and token variety."""
    words = query.split()
    word_count = len(words)
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    unique_ratio = len(set(w.lower() for w in words)) / max(1, word_count)
    score = 0.5 * min(1.0, word_count / 20.0) + 0.3 * min(1.0, avg_word_length / 10.0) + 0.2 * unique_ratio
    return max(0.0, min(1.0, score))
