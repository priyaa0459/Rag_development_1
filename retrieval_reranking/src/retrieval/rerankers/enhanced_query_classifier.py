import re
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import openai
import pickle
import os

from utils.logging_utils import get_logger, RetrievalLogger
from utils.query_utils import normalize_query, extract_query_intent
from utils.scoring_utils import calculate_openai_similarity

logger = get_logger(__name__)


class EnhancedQueryClassifier:
    """
    Enhanced query classifier with dynamic switching between multiple approaches.
    """
    
    def __init__(self, openai_client: Optional[openai.OpenAI] = None, model_path: Optional[str] = None, 
                 classification_strategy: str = 'adaptive', confidence_threshold: float = 0.5,
                 enable_cache: bool = True):
        """
        Initialize the enhanced query classifier.
        
        Args:
            openai_client: OpenAI client instance
            model_path: Path to save/load trained model
            classification_strategy: Strategy for classification ('openai', 'rule_based', 'semantic', 'adaptive')
            confidence_threshold: Minimum confidence threshold
            enable_cache: Whether to enable caching
        """
        self.client = openai_client or openai.OpenAI()
        self.model_path = model_path
        self.classification_strategy = classification_strategy
        self.confidence_threshold = confidence_threshold
        self.enable_cache = enable_cache
        
        # Cache for classification results
        self.classification_cache = {} if enable_cache else None
        
        # iFlow component patterns for rule-based classification
        self.iflow_patterns = {
            'trigger': [
                r'\b(trigger|start|initiate|begin|launch|activate|when|if|on)\b',
                r'\b(webhook|api|event|notification|schedule|timer|cron)\b',
                r'\b(monitor|watch|listen|detect|observe)\b'
            ],
            'action': [
                r'\b(action|perform|execute|run|do|send|create|update|delete)\b',
                r'\b(http|request|api|call|post|get|put|delete)\b',
                r'\b(transform|process|handle|manage|control)\b'
            ],
            'condition': [
                r'\b(condition|if|else|switch|case|when|check|validate|test)\b',
                r'\b(compare|match|filter|select|choose|decide)\b',
                r'\b(logic|rule|criteria|threshold|limit)\b'
            ],
            'transformer': [
                r'\b(transform|convert|map|format|parse|encode|decode)\b',
                r'\b(json|xml|csv|data|structure|schema)\b',
                r'\b(modify|change|update|replace|extract)\b'
            ],
            'connector': [
                r'\b(connect|link|join|bridge|route|forward|redirect)\b',
                r'\b(integration|sync|transfer|move|copy|share)\b',
                r'\b(api|service|database|file|storage)\b'
            ],
            'error_handler': [
                r'\b(error|exception|failure|retry|fallback|recovery)\b',
                r'\b(handle|catch|manage|resolve|fix|repair)\b',
                r'\b(log|alert|notify|report|monitor)\b'
            ],
            'data_mapper': [
                r'\b(map|mapping|field|column|property|attribute)\b',
                r'\b(data|record|object|entity|model|schema)\b',
                r'\b(extract|insert|update|delete|query)\b'
            ],
            'aggregator': [
                r'\b(aggregate|sum|count|average|group|collect|merge)\b',
                r'\b(batch|bulk|batch|accumulate|combine|consolidate)\b',
                r'\b(report|summary|statistics|analytics|metrics)\b'
            ]
        }
        
        # Semantic embeddings for component types
        self.component_embeddings = {
            'trigger': 'Event-driven components that start workflows based on external events or conditions',
            'action': 'Components that perform specific actions, operations, or API calls',
            'condition': 'Decision-making components that implement conditional logic and branching',
            'transformer': 'Data transformation components that convert, format, or modify data structures',
            'connector': 'Integration components that connect to external systems, APIs, or databases',
            'error_handler': 'Error handling components that manage exceptions and implement recovery logic',
            'data_mapper': 'Data mapping components that transform field mappings and data structures',
            'aggregator': 'Data aggregation components that collect, combine, and process batch data'
        }
        
        # Load trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify a query using the selected strategy.
        
        Args:
            query: Query text to classify
            
        Returns:
            Classification results
        """
        # Check cache first
        if self.enable_cache and self.classification_cache:
            cache_key = normalize_query(query)
            if cache_key in self.classification_cache:
                logger.debug(f"Using cached classification for: {query}")
                return self.classification_cache[cache_key]
        
        with RetrievalLogger(f"Classifying query: {query[:50]}...", logger):
            # Choose classification method
            if self.classification_strategy == 'openai':
                result = self._openai_classification(query)
            elif self.classification_strategy == 'rule_based':
                result = self._rule_based_classification(query)
            elif self.classification_strategy == 'semantic':
                result = self._semantic_classification(query)
            elif self.classification_strategy == 'adaptive':
                result = self._adaptive_classification(query)
            else:
                logger.warning(f"Unknown strategy: {self.classification_strategy}. Using adaptive.")
                result = self._adaptive_classification(query)
            
            # Cache result
            if self.enable_cache and self.classification_cache is not None:
                cache_key = normalize_query(query)
                self.classification_cache[cache_key] = result
            
            return result
    
    def _adaptive_classification(self, query: str) -> Dict[str, Any]:
        """
        Adaptively choose the best classification method based on query characteristics.
        
        Args:
            query: Query text
            
        Returns:
            Classification results
        """
        query_lower = query.lower()
        
        # Use OpenAI for complex queries with technical terms
        technical_terms = ['api', 'webhook', 'database', 'json', 'xml', 'http', 'authentication']
        if any(term in query_lower for term in technical_terms):
            logger.debug("Using OpenAI classification for technical query")
            return self._openai_classification(query)
        
        # Use semantic for queries with clear component descriptions
        component_terms = ['trigger', 'action', 'condition', 'transformer', 'connector', 'error', 'data', 'aggregate']
        if any(term in query_lower for term in component_terms):
            logger.debug("Using semantic classification for component-specific query")
            return self._semantic_classification(query)
        
        # Use rule-based for simple queries
        logger.debug("Using rule-based classification for simple query")
        return self._rule_based_classification(query)
    
    def _openai_classification(self, query: str) -> Dict[str, Any]:
        """
        Classify query using OpenAI.
        
        Args:
            query: Query text
            
        Returns:
            Classification results
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are an expert at classifying queries for iFlow components. 
Classify the given query into one of these component types:
- trigger: Event-driven components that start workflows
- action: Components that perform specific operations
- condition: Decision-making and conditional logic components
- transformer: Data transformation and format conversion components
- connector: Integration and connection components
- error_handler: Error handling and recovery components
- data_mapper: Data mapping and field transformation components
- aggregator: Data aggregation and batch processing components

Return a JSON object with:
- primary_type: The most likely component type
- confidence: Confidence score (0-1)
- all_types: List of all applicable types with their confidence scores
- reasoning: Brief explanation of the classification
- method: "openai"""},
                    {"role": "user", "content": f"Classify this query: {query}"}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            classification = json.loads(result_text)
            
            # Ensure required fields
            if 'primary_type' not in classification:
                classification['primary_type'] = 'action'
            if 'confidence' not in classification:
                classification['confidence'] = 0.5
            if 'all_types' not in classification:
                classification['all_types'] = {classification['primary_type']: classification['confidence']}
            if 'reasoning' not in classification:
                classification['reasoning'] = 'OpenAI classification'
            if 'method' not in classification:
                classification['method'] = 'openai'
            
            return classification
            
        except Exception as e:
            logger.warning(f"OpenAI classification failed, using fallback: {e}")
            return self._rule_based_classification(query)
    
    def _rule_based_classification(self, query: str) -> Dict[str, Any]:
        """
        Classify query using rule-based patterns.
        
        Args:
            query: Query text
            
        Returns:
            Classification results
        """
        query_lower = query.lower()
        scores = {}
        
        # Calculate scores for each component type
        for component_type, patterns in self.iflow_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches * 0.3
            
            # Normalize score
            score = min(1.0, score)
            scores[component_type] = score
        
        # Find primary type
        if scores:
            primary_type = max(scores, key=scores.get)
            confidence = scores[primary_type]
        else:
            primary_type = 'action'
            confidence = 0.1
        
        return {
            'primary_type': primary_type,
            'confidence': confidence,
            'all_types': scores,
            'reasoning': 'Rule-based pattern matching',
            'method': 'rule_based'
        }
    
    def _semantic_classification(self, query: str) -> Dict[str, Any]:
        """
        Classify query using semantic similarity with component descriptions.
        
        Args:
            query: Query text
            
        Returns:
            Classification results
        """
        scores = {}
        
        try:
            # Calculate similarity with each component description
            for component_type, description in self.component_embeddings.items():
                similarity = calculate_openai_similarity(query, description)
                scores[component_type] = similarity
            
            # Find primary type
            if scores:
                primary_type = max(scores, key=scores.get)
                confidence = scores[primary_type]
            else:
                primary_type = 'action'
                confidence = 0.1
            
            return {
                'primary_type': primary_type,
                'confidence': confidence,
                'all_types': scores,
                'reasoning': 'Semantic similarity with component descriptions',
                'method': 'semantic'
            }
            
        except Exception as e:
            logger.warning(f"Semantic classification failed, using fallback: {e}")
            return self._rule_based_classification(query)
    
    def get_component_suggestions(self, query: str) -> List[Dict[str, Any]]:
        """
        Get component suggestions based on query classification.
        
        Args:
            query: Query text
            
        Returns:
            List of component suggestions
        """
        classification = self.classify_query(query)
        primary_type = classification['primary_type']
        
        suggestions = []
        
        # Get examples for the primary type
        examples = self._get_component_examples(primary_type)
        description = self._get_component_description(primary_type)
        
        suggestions.append({
            'component': primary_type,
            'description': description,
            'examples': examples,
            'confidence': classification['confidence'],
            'classification_method': classification['method']
        })
        
        # Add other high-confidence types
        all_types = classification.get('all_types', {})
        sorted_types = sorted(all_types.items(), key=lambda x: x[1], reverse=True)
        
        for component_type, confidence in sorted_types[1:3]:  # Top 3
            if confidence > self.confidence_threshold:
                examples = self._get_component_examples(component_type)
                description = self._get_component_description(component_type)
                
                suggestions.append({
                    'component': component_type,
                    'description': description,
                    'examples': examples,
                    'confidence': confidence,
                    'classification_method': classification['method']
                })
        
        return suggestions
    
    def _get_component_description(self, component_type: str) -> str:
        """Get description for component type."""
        return self.component_embeddings.get(component_type, 'Unknown component type')
    
    def _get_component_examples(self, component_type: str) -> List[str]:
        """Get examples for component type."""
        examples = {
            'trigger': ['Webhook trigger', 'Schedule trigger', 'File watcher', 'API trigger'],
            'action': ['HTTP request', 'Email sender', 'File operation', 'Database query'],
            'condition': ['If-else logic', 'Switch statement', 'Validation check', 'Threshold comparison'],
            'transformer': ['JSON to XML', 'Data format conversion', 'Field mapping', 'Encoding/decoding'],
            'connector': ['Database connector', 'API connector', 'File system connector', 'Cloud service connector'],
            'error_handler': ['Retry logic', 'Fallback handler', 'Error logging', 'Exception recovery'],
            'data_mapper': ['Field mapping', 'Schema transformation', 'Data validation', 'Record mapping'],
            'aggregator': ['Batch processing', 'Data collection', 'Summary calculation', 'Report generation']
        }
        return examples.get(component_type, [])
    
    def set_classification_strategy(self, strategy: str) -> None:
        """
        Set the classification strategy.
        
        Args:
            strategy: New classification strategy
        """
        valid_strategies = ['openai', 'rule_based', 'semantic', 'adaptive']
        if strategy not in valid_strategies:
            logger.warning(f"Invalid strategy: {strategy}. Using 'adaptive'")
            strategy = 'adaptive'
        
        self.classification_strategy = strategy
        logger.info(f"Classification strategy set to: {strategy}")
    
    def clear_cache(self) -> None:
        """Clear the classification cache."""
        if self.classification_cache:
            self.classification_cache.clear()
            logger.info("Classification cache cleared")
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """
        Get classification statistics.
        
        Returns:
            Classification statistics
        """
        return {
            'classification_strategy': self.classification_strategy,
            'confidence_threshold': self.confidence_threshold,
            'supported_types': list(self.iflow_patterns.keys()),
            'pattern_count': sum(len(patterns) for patterns in self.iflow_patterns.values()),
            'cache_enabled': self.enable_cache,
            'cache_size': len(self.classification_cache) if self.classification_cache else 0
        }
    
    def save_model(self, model_path: Optional[str] = None) -> None:
        """
        Save classifier configuration.
        
        Args:
            model_path: Path to save model
        """
        model_path = model_path or self.model_path
        if not model_path:
            logger.warning("No model path provided")
            return
        
        with RetrievalLogger("Saving enhanced query classifier", logger):
            config = {
                'classification_strategy': self.classification_strategy,
                'confidence_threshold': self.confidence_threshold,
                'iflow_patterns': self.iflow_patterns,
                'component_embeddings': self.component_embeddings,
                'enable_cache': self.enable_cache
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(config, f)
            
            logger.info(f"Saved enhanced classifier configuration to {model_path}")
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load classifier configuration.
        
        Args:
            model_path: Path to load model from
        """
        model_path = model_path or self.model_path
        if not model_path or not os.path.exists(model_path):
            logger.warning("Model file not found")
            return
        
        with RetrievalLogger("Loading enhanced query classifier", logger):
            try:
                with open(model_path, 'rb') as f:
                    config = pickle.load(f)
                
                self.classification_strategy = config.get('classification_strategy', self.classification_strategy)
                self.confidence_threshold = config.get('confidence_threshold', self.confidence_threshold)
                self.iflow_patterns = config.get('iflow_patterns', self.iflow_patterns)
                self.component_embeddings = config.get('component_embeddings', self.component_embeddings)
                self.enable_cache = config.get('enable_cache', self.enable_cache)
                
                logger.info(f"Loaded enhanced classifier configuration from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load enhanced classifier: {e}")
