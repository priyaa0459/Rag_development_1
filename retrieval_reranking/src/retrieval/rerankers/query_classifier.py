import re
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np
import openai
import pickle
import os

from utils.logging_utils import get_logger, RetrievalLogger
from utils.query_utils import normalize_query, extract_query_intent

logger = get_logger(__name__)


class QueryClassifier:
    """
    Query classifier for iFlow component types using OpenAI.
    """
    
    def __init__(self, openai_client: Optional[openai.OpenAI] = None, model_path: Optional[str] = None, classifier_type: str = 'openai', confidence_threshold: float = 0.5):
        """
        Initialize the query classifier.
        
        Args:
            openai_client: OpenAI client instance
            model_path: Path to save/load trained model
            classifier_type: Type of classifier ('openai' or 'rule_based')
            confidence_threshold: Minimum confidence threshold
        """
        self.client = openai_client or openai.OpenAI()
        self.model_path = model_path
        self.classifier_type = classifier_type
        self.confidence_threshold = confidence_threshold
        
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
        
        # Load trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        logger.info(f"Query classifier initialized with type: {classifier_type}")
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify query into iFlow component types.
        
        Args:
            query: Query text to classify
            
        Returns:
            Classification results with component types and confidence scores
        """
        if self.classifier_type == 'openai':
            return self._openai_classification(query)
        else:
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
                    {"role": "system", "content": """You are an iFlow component classifier. Classify the given query into one or more iFlow component types from this list:
- trigger: Event-driven components that start workflows
- action: Components that perform specific actions or operations
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
- reasoning: Brief explanation of the classification"""},
                    {"role": "user", "content": f"Classify this query: {query}"}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            classification = json.loads(result_text)
            
            # Ensure required fields
            if 'primary_type' not in classification:
                classification['primary_type'] = 'action'  # Default
            if 'confidence' not in classification:
                classification['confidence'] = 0.5
            if 'all_types' not in classification:
                classification['all_types'] = {classification['primary_type']: classification['confidence']}
            if 'reasoning' not in classification:
                classification['reasoning'] = 'OpenAI classification'
            
            return classification
            
        except Exception as e:
            logger.warning(f"OpenAI classification failed, using rule-based fallback: {e}")
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
                score += matches * 0.3  # Weight for each match
            
            # Normalize score
            score = min(1.0, score)
            scores[component_type] = score
        
        # Find primary type
        if scores:
            primary_type = max(scores, key=scores.get)
            confidence = scores[primary_type]
        else:
            primary_type = 'action'  # Default
            confidence = 0.1
        
        return {
            'primary_type': primary_type,
            'confidence': confidence,
            'all_types': scores,
            'reasoning': 'Rule-based pattern matching'
        }
    
    def train_model(self, training_data: List[Tuple[str, str]], test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train the classifier (not applicable for OpenAI-based classifier).
        
        Args:
            training_data: List of (query, label) pairs
            test_size: Fraction of data for testing
            random_state: Random seed
            
        Returns:
            Training results
        """
        logger.info("Training not applicable for OpenAI-based classifier")
        return {
            'accuracy': 1.0,
            'model_type': 'openai',
            'training_samples': len(training_data),
            'message': 'OpenAI-based classifier does not require training'
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
        
        with RetrievalLogger("Saving query classifier", logger):
            config = {
                'classifier_type': self.classifier_type,
                'confidence_threshold': self.confidence_threshold,
                'iflow_patterns': self.iflow_patterns
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(config, f)
            
            logger.info(f"Saved classifier configuration to {model_path}")
    
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
        
        with RetrievalLogger("Loading query classifier", logger):
            try:
                with open(model_path, 'rb') as f:
                    config = pickle.load(f)
                
                self.classifier_type = config.get('classifier_type', self.classifier_type)
                self.confidence_threshold = config.get('confidence_threshold', self.confidence_threshold)
                self.iflow_patterns = config.get('iflow_patterns', self.iflow_patterns)
                
                logger.info(f"Loaded classifier configuration from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load classifier: {e}")
    
    def get_component_suggestions(self, query: str) -> List[Dict[str, Any]]:
        """
        Get component suggestions based on query classification.
        
        Args:
            query: Query text
            
        Returns:
            List of component suggestions
        """
        classification = self.classify_query(query)
        
        suggestions = []
        all_types = classification.get('all_types', [])
        
        # Handle both list and dict formats for all_types
        if isinstance(all_types, list):
            # If it's a list, use the primary type and default confidence
            primary_type = classification.get('primary_type', 'action')
            confidence = classification.get('confidence', 0.5)
            
            if confidence >= self.confidence_threshold:
                suggestion = {
                    'component_type': primary_type,
                    'confidence': confidence,
                    'description': self._get_component_description(primary_type),
                    'examples': self._get_component_examples(primary_type)
                }
                suggestions.append(suggestion)
        else:
            # If it's a dict, process each type
            for component_type, confidence in all_types.items():
                if confidence >= self.confidence_threshold:
                    suggestion = {
                        'component_type': component_type,
                        'confidence': confidence,
                        'description': self._get_component_description(component_type),
                        'examples': self._get_component_examples(component_type)
                    }
                    suggestions.append(suggestion)
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return suggestions
    
    def _get_component_description(self, component_type: str) -> str:
        """Get description for component type."""
        descriptions = {
            'trigger': 'Event-driven components that start workflows based on external events or conditions',
            'action': 'Components that perform specific actions, operations, or API calls',
            'condition': 'Decision-making components that implement conditional logic and branching',
            'transformer': 'Data transformation components that convert, format, or modify data structures',
            'connector': 'Integration components that connect to external systems, APIs, or databases',
            'error_handler': 'Error handling components that manage exceptions and implement recovery logic',
            'data_mapper': 'Data mapping components that transform field mappings and data structures',
            'aggregator': 'Data aggregation components that collect, combine, and process batch data'
        }
        return descriptions.get(component_type, 'Unknown component type')
    
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
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """
        Get classification statistics.
        
        Returns:
            Classification statistics
        """
        return {
            'classifier_type': self.classifier_type,
            'confidence_threshold': self.confidence_threshold,
            'supported_types': list(self.iflow_patterns.keys()),
            'pattern_count': sum(len(patterns) for patterns in self.iflow_patterns.values())
        }
