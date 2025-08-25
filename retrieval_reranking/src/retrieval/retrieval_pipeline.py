from typing import List, Dict, Any, Optional, Tuple
import json
import os
from pathlib import Path

from utils.logging_utils import get_logger, RetrievalLogger
from retrieval.vector_search import VectorSearch
from retrieval.hybrid_search import HybridSearch
from retrieval.rerankers.cross_encoder_reranker import CrossEncoderReranker
from retrieval.rerankers.metadata_reranker import MetadataReranker
from retrieval.rerankers.query_classifier import QueryClassifier
from retrieval.rerankers.hybrid_reranker import HybridReranker

logger = get_logger(__name__)


class RetrievalPipeline:
    """
    Main retrieval and reranking pipeline orchestrating all components.
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None, index_path: Optional[str] = None, enable_cross_encoder: bool = True, enable_metadata_reranking: bool = True, enable_query_classification: bool = True, enable_hybrid_reranking: bool = True, log_file: Optional[str] = None):
        """
        Initialize the retrieval pipeline.
        
        Args:
            model_config: Configuration for models and components
            index_path: Path for saving/loading indices
            enable_cross_encoder: Whether to enable cross-encoder reranking
            enable_metadata_reranking: Whether to enable metadata reranking
            enable_query_classification: Whether to enable query classification
            enable_hybrid_reranking: Whether to enable hybrid reranking
            log_file: Path to log file
        """
        # Default configuration
        self.default_config = {
            'vector_search': {
                'model_name': 'text-embedding-ada-002',
                'dimension': 1536
            },
            'cross_encoder': {
                'model_name': 'gpt-3.5-turbo',
                'max_length': 512
            },
            'query_classifier': {
                'model_name': 'gpt-3.5-turbo',
                'classifier_type': 'openai'
            }
        }
        
        # Merge with provided config
        self.config = self._merge_config(model_config or {})
        
        # Pipeline settings
        self.index_path = index_path or "./data"
        self.enable_cross_encoder = enable_cross_encoder
        self.enable_metadata_reranking = enable_metadata_reranking
        self.enable_query_classification = enable_query_classification
        self.enable_hybrid_reranking = enable_hybrid_reranking
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Retrieval pipeline initialized")
    
    def _merge_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user configuration with defaults."""
        merged = self.default_config.copy()
        
        for section, config in user_config.items():
            if section in merged:
                merged[section].update(config)
            else:
                merged[section] = config
        
        return merged
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        with RetrievalLogger("Initializing retrieval pipeline components", logger):
            # Create index directory
            os.makedirs(self.index_path, exist_ok=True)
            
            # Initialize vector search
            vector_config = self.config['vector_search']
            self.vector_search = VectorSearch(
                model_name=vector_config['model_name'],
                index_path=f"{self.index_path}/faiss_index",
                documents_path=f"{self.index_path}/documents.pkl",
                dimension=vector_config['dimension']
            )
            
            # Initialize rerankers
            self.cross_encoder_reranker = None
            self.metadata_reranker = None
            self.query_classifier = None
            self.hybrid_reranker = None
            
            if self.enable_cross_encoder:
                cross_encoder_config = self.config['cross_encoder']
                self.cross_encoder_reranker = CrossEncoderReranker(
                    model_name=cross_encoder_config['model_name'],
                    max_length=cross_encoder_config['max_length']
                )
            
            if self.enable_metadata_reranking:
                self.metadata_reranker = MetadataReranker()
            
            if self.enable_query_classification:
                classifier_config = self.config['query_classifier']
                self.query_classifier = QueryClassifier(
                    classifier_type=classifier_config['classifier_type']
                )
            
            if self.enable_hybrid_reranking:
                self.hybrid_reranker = HybridReranker(
                    cross_encoder_reranker=self.cross_encoder_reranker,
                    metadata_reranker=self.metadata_reranker,
                    query_classifier=self.query_classifier
                )
            
            # Initialize hybrid search
            self.hybrid_search = HybridSearch(
                vector_search=self.vector_search,
                hybrid_reranker=self.hybrid_reranker
            )
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the retrieval pipeline.
        
        Args:
            documents: List of documents to add
        """
        with RetrievalLogger(f"Adding {len(documents)} documents to pipeline", logger):
            self.vector_search.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to retrieval pipeline")
    
    def search(self, query: str, top_k: int = 10, threshold: float = 0.0, enable_reranking: bool = True, return_breakdown: bool = False) -> Dict[str, Any]:
        """
        Perform search with optional reranking.
        
        Args:
            query: Search query
            top_k: Number of top results
            threshold: Minimum similarity threshold
            enable_reranking: Whether to apply reranking
            return_breakdown: Whether to return score breakdown
            
        Returns:
            Search results with metadata
        """
        with RetrievalLogger(f"Pipeline search for query: {query[:50]}...", logger):
            # Perform hybrid search
            search_results = self.hybrid_search.search(
                query=query,
                top_k=top_k,
                threshold=threshold,
                enable_reranking=enable_reranking
            )
            
            # Get component suggestions if classifier is available
            component_suggestions = []
            if self.query_classifier:
                component_suggestions = self.query_classifier.get_component_suggestions(query)
            
            # Prepare response
            response = {
                'query': query,
                'results': search_results,
                'total_results': len(search_results),
                'component_suggestions': component_suggestions,
                'pipeline_config': {
                    'enable_cross_encoder': self.enable_cross_encoder,
                    'enable_metadata_reranking': self.enable_metadata_reranking,
                    'enable_query_classification': self.enable_query_classification,
                    'enable_hybrid_reranking': self.enable_hybrid_reranking
                }
            }
            
            if return_breakdown and search_results:
                # Add score breakdown for first result
                first_result = search_results[0]
                response['score_breakdown'] = {
                    'vector_score': first_result.get('similarity_score', 0.0),
                    'cross_encoder_score': first_result.get('cross_encoder_score', 0.0),
                    'metadata_score': first_result.get('metadata_score', 0.0),
                    'final_score': first_result.get('final_score', 0.0)
                }
            
            return response
    
    def batch_search(self, queries: List[str], top_k: int = 10, threshold: float = 0.0, enable_reranking: bool = True) -> List[Dict[str, Any]]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of top results per query
            threshold: Minimum similarity threshold
            enable_reranking: Whether to apply reranking
            
        Returns:
            List of search results for each query
        """
        with RetrievalLogger(f"Batch search for {len(queries)} queries", logger):
            results = []
            for query in queries:
                query_result = self.search(query, top_k, threshold, enable_reranking)
                results.append(query_result)
            return results
    
    def get_component_suggestions(self, query: str) -> List[Dict[str, Any]]:
        """
        Get iFlow component suggestions for a query.
        
        Args:
            query: Search query
            
        Returns:
            List of component suggestions
        """
        if not self.query_classifier:
            return []
        
        return self.query_classifier.get_component_suggestions(query)
    
    def train_query_classifier(self, training_data: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Train the query classifier.
        
        Args:
            training_data: List of (query, label) pairs
            
        Returns:
            Training results
        """
        if not self.query_classifier:
            return {'error': 'Query classifier not enabled'}
        
        return self.query_classifier.train_model(training_data)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the pipeline and its components.
        
        Returns:
            Pipeline statistics
        """
        stats = {
            'vector_search': self.vector_search.get_index_stats(),
            'pipeline_config': {
                'enable_cross_encoder': self.enable_cross_encoder,
                'enable_metadata_reranking': self.enable_metadata_reranking,
                'enable_query_classification': self.enable_query_classification,
                'enable_hybrid_reranking': self.enable_hybrid_reranking
            },
            'components': {
                'cross_encoder_reranker': self.cross_encoder_reranker is not None,
                'metadata_reranker': self.metadata_reranker is not None,
                'query_classifier': self.query_classifier is not None,
                'hybrid_reranker': self.hybrid_reranker is not None
            }
        }
        
        # Add component-specific stats
        if self.cross_encoder_reranker:
            stats['cross_encoder'] = self.cross_encoder_reranker.get_model_info()
        
        if self.query_classifier:
            stats['query_classifier'] = self.query_classifier.get_classification_stats()
        
        if self.hybrid_reranker:
            stats['hybrid_reranker'] = self.hybrid_reranker.get_reranker_stats()
        
        return stats
    
    def save_pipeline(self, save_path: Optional[str] = None) -> None:
        """
        Save the pipeline state and components.
        
        Args:
            save_path: Path to save pipeline state
        """
        save_path = save_path or f"{self.index_path}/pipeline_state.json"
        
        with RetrievalLogger("Saving pipeline state", logger):
            # Save vector search index
            self.vector_search.save_index()
            
            # Save query classifier if available
            if self.query_classifier:
                classifier_path = f"{self.index_path}/query_classifier.pkl"
                self.query_classifier.save_model(classifier_path)
            
            # Save pipeline configuration
            pipeline_state = {
                'config': self.config,
                'pipeline_settings': {
                    'enable_cross_encoder': self.enable_cross_encoder,
                    'enable_metadata_reranking': self.enable_metadata_reranking,
                    'enable_query_classification': self.enable_query_classification,
                    'enable_hybrid_reranking': self.enable_hybrid_reranking
                }
            }
            
            with open(save_path, 'w') as f:
                json.dump(pipeline_state, f, indent=2)
            
            logger.info(f"Pipeline state saved to {save_path}")
    
    def load_pipeline(self, load_path: Optional[str] = None) -> None:
        """
        Load the pipeline state and components.
        
        Args:
            load_path: Path to load pipeline state from
        """
        load_path = load_path or f"{self.index_path}/pipeline_state.json"
        
        if not os.path.exists(load_path):
            logger.warning(f"Pipeline state file not found: {load_path}")
            return
        
        with RetrievalLogger("Loading pipeline state", logger):
            # Load pipeline configuration
            with open(load_path, 'r') as f:
                pipeline_state = json.load(f)
            
            # Update configuration
            self.config = pipeline_state.get('config', self.config)
            settings = pipeline_state.get('pipeline_settings', {})
            self.enable_cross_encoder = settings.get('enable_cross_encoder', self.enable_cross_encoder)
            self.enable_metadata_reranking = settings.get('enable_metadata_reranking', self.enable_metadata_reranking)
            self.enable_query_classification = settings.get('enable_query_classification', self.enable_query_classification)
            self.enable_hybrid_reranking = settings.get('enable_hybrid_reranking', self.enable_hybrid_reranking)
            
            # Reinitialize components with loaded config
            self._initialize_components()
            
            # Load vector search index
            self.vector_search.load_index()
            
            # Load query classifier if available
            if self.query_classifier:
                classifier_path = f"{self.index_path}/query_classifier.pkl"
                self.query_classifier.load_model(classifier_path)
            
            logger.info(f"Pipeline state loaded from {load_path}")
    
    def clear_pipeline(self) -> None:
        """Clear all documents and reset the pipeline."""
        with RetrievalLogger("Clearing pipeline", logger):
            self.vector_search.clear_index()
            logger.info("Pipeline cleared")
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update pipeline configuration.
        
        Args:
            new_config: New configuration to merge
        """
        self.config = self._merge_config(new_config)
        logger.info("Pipeline configuration updated")
    
    def enable_component(self, component: str, enable: bool = True) -> None:
        """
        Enable or disable a pipeline component.
        
        Args:
            component: Component name ('cross_encoder', 'metadata_reranking', 'query_classification', 'hybrid_reranking')
            enable: Whether to enable the component
        """
        if component == 'cross_encoder':
            self.enable_cross_encoder = enable
        elif component == 'metadata_reranking':
            self.enable_metadata_reranking = enable
        elif component == 'query_classification':
            self.enable_query_classification = enable
        elif component == 'hybrid_reranking':
            self.enable_hybrid_reranking = enable
        else:
            logger.warning(f"Unknown component: {component}")
            return
        
        # Reinitialize components if needed
        if enable:
            self._initialize_components()
        
        logger.info(f"{'Enabled' if enable else 'Disabled'} component: {component}")
