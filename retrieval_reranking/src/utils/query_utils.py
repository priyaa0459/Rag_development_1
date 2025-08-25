import re
import string
from typing import List, Dict, Any, Optional
from collections import Counter
import openai
import tiktoken

from utils.logging_utils import get_logger

logger = get_logger(__name__)

class QueryProcessor:
    """OpenAI-based query processor for text preprocessing and analysis."""
    
    def __init__(self, openai_client: Optional[openai.OpenAI] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the query processor.
        
        Args:
            openai_client: OpenAI client instance
            model: OpenAI model to use for text processing
        """
        self.client = openai_client or openai.OpenAI()
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess query text using OpenAI.
        
        Args:
            query: Raw query text
            
        Returns:
            Preprocessed query text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a text preprocessing assistant. Clean and normalize the given text by removing extra whitespace, converting to lowercase, and removing special characters while preserving meaning."},
                    {"role": "user", "content": f"Preprocess this text: {query}"}
                ],
                max_tokens=100,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI preprocessing failed, using fallback: {e}")
            return self._fallback_preprocess(query)
    
    def _fallback_preprocess(self, query: str) -> str:
        """Fallback preprocessing without OpenAI."""
        # Remove extra whitespace and convert to lowercase
        query = re.sub(r'\s+', ' ', query.strip().lower())
        # Remove special characters but keep alphanumeric and spaces
        query = re.sub(r'[^\w\s]', '', query)
        return query
    
    def tokenize_query(self, query: str) -> List[str]:
        """
        Tokenize query using OpenAI's tokenizer.
        
        Args:
            query: Query text
            
        Returns:
            List of tokens
        """
        try:
            tokens = self.encoding.encode(query)
            return [self.encoding.decode([token]) for token in tokens if token != self.encoding.encode(' ')[0]]
        except Exception as e:
            logger.warning(f"OpenAI tokenization failed, using fallback: {e}")
            return self._fallback_tokenize(query)
    
    def _fallback_tokenize(self, query: str) -> List[str]:
        """Fallback tokenization without OpenAI."""
        return query.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords using OpenAI.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Tokens with stopwords removed
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a text processing assistant. Remove common stopwords from the given list of words. Return only the meaningful words as a comma-separated list."},
                    {"role": "user", "content": f"Remove stopwords from: {', '.join(tokens)}"}
                ],
                max_tokens=200,
                temperature=0.1
            )
            result = response.choices[0].message.content.strip()
            return [word.strip() for word in result.split(',') if word.strip()]
        except Exception as e:
            logger.warning(f"OpenAI stopword removal failed, using fallback: {e}")
            return self._fallback_remove_stopwords(tokens)
    
    def _fallback_remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Fallback stopword removal without OpenAI."""
        # Basic English stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        return [token for token in tokens if token.lower() not in stopwords]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens using OpenAI.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized tokens
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a text processing assistant. Convert words to their base form (lemmatize). Return the lemmatized words as a comma-separated list."},
                    {"role": "user", "content": f"Lemmatize these words: {', '.join(tokens)}"}
                ],
                max_tokens=200,
                temperature=0.1
            )
            result = response.choices[0].message.content.strip()
            return [word.strip() for word in result.split(',') if word.strip()]
        except Exception as e:
            logger.warning(f"OpenAI lemmatization failed, using fallback: {e}")
            return tokens  # Return original tokens as fallback
    
    def get_query_features(self, query: str) -> Dict[str, Any]:
        """
        Extract comprehensive features from query using OpenAI.
        
        Args:
            query: Query text
            
        Returns:
            Dictionary of query features
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a query analysis assistant. Analyze the given query and return a JSON object with the following features: tokens (list), word_count (int), avg_word_length (float), complexity_score (float 0-1), keywords (list), intent (string), and domain (string)."},
                    {"role": "user", "content": f"Analyze this query: {query}"}
                ],
                max_tokens=300,
                temperature=0.1
            )
            import json
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            logger.warning(f"OpenAI feature extraction failed, using fallback: {e}")
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
        """
        Extract keywords from query using OpenAI.
        
        Args:
            query: Query text
            top_k: Number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a keyword extraction assistant. Extract the top {top_k} most important keywords from the given text. Return them as a comma-separated list."},
                    {"role": "user", "content": f"Extract keywords from: {query}"}
                ],
                max_tokens=100,
                temperature=0.1
            )
            result = response.choices[0].message.content.strip()
            return [word.strip() for word in result.split(',')[:top_k] if word.strip()]
        except Exception as e:
            logger.warning(f"OpenAI keyword extraction failed, using fallback: {e}")
            return self._fallback_extract_keywords(query, top_k)
    
    def _fallback_extract_keywords(self, query: str, top_k: int = 5) -> List[str]:
        """Fallback keyword extraction without OpenAI."""
        tokens = self.tokenize_query(query)
        # Simple frequency-based keyword extraction
        word_freq = Counter(tokens)
        return [word for word, _ in word_freq.most_common(top_k)]


def normalize_query(query: str) -> str:
    """
    Normalize query text using OpenAI.
    
    Args:
        query: Raw query text
        
    Returns:
        Normalized query text
    """
    processor = QueryProcessor()
    return processor.preprocess_query(query)


def extract_query_intent(query: str) -> Dict[str, Any]:
    """
    Extract query intent using OpenAI.
    
    Args:
        query: Query text
        
    Returns:
        Dictionary with intent information
    """
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a query intent analysis assistant. Analyze the given query and return a JSON object with: intent (string), confidence (float 0-1), entities (list), and action_type (string)."},
                {"role": "user", "content": f"Analyze the intent of: {query}"}
            ],
            max_tokens=200,
            temperature=0.1
        )
        import json
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        logger.warning(f"OpenAI intent extraction failed, using fallback: {e}")
        return {
            'intent': 'search',
            'confidence': 0.5,
            'entities': [],
            'action_type': 'query'
        }


def calculate_query_complexity(query: str) -> float:
    """
    Calculate query complexity using OpenAI.
    
    Args:
        query: Query text
        
    Returns:
        Complexity score between 0 and 1
    """
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a query complexity analyzer. Rate the complexity of the given query on a scale from 0 to 1, where 0 is very simple and 1 is very complex. Consider factors like vocabulary, sentence structure, and technical terms. Return only the numeric score."},
                {"role": "user", "content": f"Rate the complexity of: {query}"}
            ],
            max_tokens=10,
            temperature=0.1
        )
        score = float(response.choices[0].message.content.strip())
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    except Exception as e:
        logger.warning(f"OpenAI complexity calculation failed, using fallback: {e}")
        # Fallback complexity calculation
        words = query.split()
        word_count = len(words)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        return min(1.0, (word_count * avg_word_length) / 50.0)  # Simple heuristic
