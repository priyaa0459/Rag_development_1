import numpy as np
import json
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from utils.logging_utils import get_logger, RetrievalLogger
from utils.scoring_utils import normalize_scores, calculate_hybrid_score

logger = get_logger(__name__)


class TrainableHybridReranker:
    """
    Trainable hybrid reranker that learns optimal fusion weights for combining scoring signals.
    """
    
    def __init__(self, model_path: Optional[str] = None, fusion_method: str = 'logistic_regression', 
                 enable_online_learning: bool = True, learning_rate: float = 0.01):
        """
        Initialize the trainable hybrid reranker.
        
        Args:
            model_path: Path to save/load trained model
            fusion_method: Fusion method ('logistic_regression', 'linear_regression', 'neural_network', 'weighted_sum')
            enable_online_learning: Whether to enable online learning
            learning_rate: Learning rate for online updates
        """
        self.model_path = model_path
        self.fusion_method = fusion_method
        self.enable_online_learning = enable_online_learning
        self.learning_rate = learning_rate
        
        # Initialize fusion model
        self.fusion_model = None
        self.feature_names = ['vector_score', 'cross_encoder_score', 'metadata_score', 'classification_score']
        self.n_features = len(self.feature_names)
        
        # Default weights (used as fallback)
        self.default_weights = {
            'vector_score': 0.4,
            'cross_encoder_score': 0.3,
            'metadata_score': 0.2,
            'classification_score': 0.1
        }
        
        # Training data
        self.training_data = []
        self.validation_data = []
        
        # Load trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._initialize_model()
        
        logger.info(f"Trainable hybrid reranker initialized with fusion method: {fusion_method}")
    
    def _initialize_model(self) -> None:
        """Initialize the fusion model based on the selected method."""
        if self.fusion_method == 'logistic_regression':
            # Simple logistic regression implementation
            self.fusion_model = {
                'weights': np.random.randn(self.n_features) * 0.1,
                'bias': 0.0,
                'method': 'logistic_regression'
            }
        elif self.fusion_method == 'linear_regression':
            # Simple linear regression implementation
            self.fusion_model = {
                'weights': np.random.randn(self.n_features) * 0.1,
                'bias': 0.0,
                'method': 'linear_regression'
            }
        elif self.fusion_method == 'neural_network':
            # Simple neural network implementation
            hidden_size = 8
            self.fusion_model = {
                'weights1': np.random.randn(self.n_features, hidden_size) * 0.1,
                'bias1': np.zeros(hidden_size),
                'weights2': np.random.randn(hidden_size, 1) * 0.1,
                'bias2': 0.0,
                'method': 'neural_network'
            }
        elif self.fusion_method == 'weighted_sum':
            # Simple weighted sum (no training needed)
            self.fusion_model = {
                'weights': np.array(list(self.default_weights.values())),
                'method': 'weighted_sum'
            }
        else:
            logger.warning(f"Unknown fusion method: {self.fusion_method}. Using weighted_sum.")
            self.fusion_method = 'weighted_sum'
            self._initialize_model()
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               vector_scores: List[float], cross_encoder_scores: List[float], 
               metadata_scores: List[float], classification_scores: List[float],
               top_k: Optional[int] = None, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Rerank documents using the trained fusion model.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            vector_scores: Vector similarity scores
            cross_encoder_scores: Cross-encoder scores
            metadata_scores: Metadata-based scores
            classification_scores: Classification-based scores
            top_k: Number of top results to return
            threshold: Minimum score threshold
            
        Returns:
            Reranked documents with scores
        """
        if not documents:
            return []
        
        with RetrievalLogger(f"Trainable hybrid reranking for {len(documents)} documents", logger):
            # Prepare features
            features = self._prepare_features(vector_scores, cross_encoder_scores, 
                                            metadata_scores, classification_scores)
            
            # Get fusion scores
            fusion_scores = self._get_fusion_scores(features)
            
            # Combine documents with scores
            results = []
            for i, (doc, fusion_score) in enumerate(zip(documents, fusion_scores)):
                if fusion_score >= threshold:
                    result = doc.copy()
                    result['vector_score'] = vector_scores[i]
                    result['cross_encoder_score'] = cross_encoder_scores[i]
                    result['metadata_score'] = metadata_scores[i]
                    result['classification_score'] = classification_scores[i]
                    result['fusion_score'] = fusion_score
                    result['final_score'] = fusion_score
                    results.append(result)
            
            # Sort by fusion score (descending)
            results.sort(key=lambda x: x['fusion_score'], reverse=True)
            
            # Apply top_k limit
            if top_k is not None:
                results = results[:top_k]
            
            logger.info(f"Reranked {len(results)} documents above threshold {threshold}")
            return results
    
    def _prepare_features(self, vector_scores: List[float], cross_encoder_scores: List[float],
                         metadata_scores: List[float], classification_scores: List[float]) -> np.ndarray:
        """
        Prepare feature matrix for fusion.
        
        Args:
            vector_scores: Vector similarity scores
            cross_encoder_scores: Cross-encoder scores
            metadata_scores: Metadata-based scores
            classification_scores: Classification-based scores
            
        Returns:
            Feature matrix
        """
        # Normalize scores
        vector_scores_norm = normalize_scores(vector_scores, method='minmax')
        cross_encoder_scores_norm = normalize_scores(cross_encoder_scores, method='minmax')
        metadata_scores_norm = normalize_scores(metadata_scores, method='minmax')
        classification_scores_norm = normalize_scores(classification_scores, method='minmax')
        
        # Create feature matrix
        features = np.column_stack([
            vector_scores_norm,
            cross_encoder_scores_norm,
            metadata_scores_norm,
            classification_scores_norm
        ])
        
        return features
    
    def _get_fusion_scores(self, features: np.ndarray) -> List[float]:
        """
        Get fusion scores using the trained model.
        
        Args:
            features: Feature matrix
            
        Returns:
            List of fusion scores
        """
        if self.fusion_method == 'logistic_regression':
            return self._logistic_regression_predict(features)
        elif self.fusion_method == 'linear_regression':
            return self._linear_regression_predict(features)
        elif self.fusion_method == 'neural_network':
            return self._neural_network_predict(features)
        elif self.fusion_method == 'weighted_sum':
            return self._weighted_sum_predict(features)
        else:
            return self._weighted_sum_predict(features)
    
    def _logistic_regression_predict(self, features: np.ndarray) -> List[float]:
        """Predict using logistic regression."""
        weights = self.fusion_model['weights']
        bias = self.fusion_model['bias']
        
        # Linear combination
        logits = np.dot(features, weights) + bias
        
        # Sigmoid activation
        scores = 1 / (1 + np.exp(-logits))
        
        return scores.flatten().tolist()
    
    def _linear_regression_predict(self, features: np.ndarray) -> List[float]:
        """Predict using linear regression."""
        weights = self.fusion_model['weights']
        bias = self.fusion_model['bias']
        
        # Linear combination
        scores = np.dot(features, weights) + bias
        
        # Clip to [0, 1]
        scores = np.clip(scores, 0, 1)
        
        return scores.flatten().tolist()
    
    def _neural_network_predict(self, features: np.ndarray) -> List[float]:
        """Predict using neural network."""
        weights1 = self.fusion_model['weights1']
        bias1 = self.fusion_model['bias1']
        weights2 = self.fusion_model['weights2']
        bias2 = self.fusion_model['bias2']
        
        # Forward pass
        hidden = np.dot(features, weights1) + bias1
        hidden = np.maximum(hidden, 0)  # ReLU activation
        
        output = np.dot(hidden, weights2) + bias2
        
        # Sigmoid activation for final output
        scores = 1 / (1 + np.exp(-output))
        
        return scores.flatten().tolist()
    
    def _weighted_sum_predict(self, features: np.ndarray) -> List[float]:
        """Predict using weighted sum."""
        weights = self.fusion_model['weights']
        
        # Weighted sum
        scores = np.dot(features, weights)
        
        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores.tolist()
    
    def train(self, training_data: List[Dict[str, Any]], validation_data: Optional[List[Dict[str, Any]]] = None,
              epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Train the fusion model.
        
        Args:
            training_data: List of training examples
            validation_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            Training results
        """
        with RetrievalLogger("Training fusion model", logger):
            if not training_data:
                logger.warning("No training data provided")
                return {'error': 'No training data'}
            
            # Prepare training data
            X_train, y_train = self._prepare_training_data(training_data)
            
            # Prepare validation data
            X_val, y_val = None, None
            if validation_data:
                X_val, y_val = self._prepare_training_data(validation_data)
            
            # Training history
            history = {
                'train_loss': [],
                'val_loss': [],
                'train_accuracy': [],
                'val_accuracy': []
            }
            
            # Training loop
            for epoch in range(epochs):
                # Shuffle training data
                indices = np.random.permutation(len(X_train))
                X_train_shuffled = X_train[indices]
                y_train_shuffled = y_train[indices]
                
                # Mini-batch training
                for i in range(0, len(X_train_shuffled), batch_size):
                    batch_X = X_train_shuffled[i:i+batch_size]
                    batch_y = y_train_shuffled[i:i+batch_size]
                    
                    # Update model
                    self._update_model(batch_X, batch_y, learning_rate)
                
                # Calculate metrics
                train_loss = self._calculate_loss(X_train, y_train)
                train_accuracy = self._calculate_accuracy(X_train, y_train)
                
                history['train_loss'].append(train_loss)
                history['train_accuracy'].append(train_accuracy)
                
                if X_val is not None:
                    val_loss = self._calculate_loss(X_val, y_val)
                    val_accuracy = self._calculate_accuracy(X_val, y_val)
                    
                    history['val_loss'].append(val_loss)
                    history['val_accuracy'].append(val_accuracy)
                
                # Log progress
                if epoch % 10 == 0:
                    log_msg = f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}"
                    if X_val is not None:
                        log_msg += f", val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}"
                    logger.info(log_msg)
            
            # Save model
            if self.model_path:
                self.save_model()
            
            return {
                'epochs': epochs,
                'final_train_loss': history['train_loss'][-1],
                'final_train_accuracy': history['train_accuracy'][-1],
                'history': history
            }
    
    def _prepare_training_data(self, data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for the fusion model.
        
        Args:
            data: List of training examples
            
        Returns:
            Tuple of (features, labels)
        """
        features_list = []
        labels_list = []
        
        for example in data:
            # Extract features
            features = [
                example.get('vector_score', 0.0),
                example.get('cross_encoder_score', 0.0),
                example.get('metadata_score', 0.0),
                example.get('classification_score', 0.0)
            ]
            
            # Extract label (relevance score)
            label = example.get('relevance_score', 0.0)
            
            features_list.append(features)
            labels_list.append(label)
        
        return np.array(features_list), np.array(labels_list)
    
    def _update_model(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        """
        Update the fusion model parameters.
        
        Args:
            X: Feature matrix
            y: Labels
            learning_rate: Learning rate
        """
        if self.fusion_method == 'logistic_regression':
            self._update_logistic_regression(X, y, learning_rate)
        elif self.fusion_method == 'linear_regression':
            self._update_linear_regression(X, y, learning_rate)
        elif self.fusion_method == 'neural_network':
            self._update_neural_network(X, y, learning_rate)
    
    def _update_logistic_regression(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        """Update logistic regression parameters."""
        weights = self.fusion_model['weights']
        bias = self.fusion_model['bias']
        
        # Forward pass
        logits = np.dot(X, weights) + bias
        predictions = 1 / (1 + np.exp(-logits))
        
        # Gradients
        dw = np.dot(X.T, (predictions - y)) / len(X)
        db = np.mean(predictions - y)
        
        # Update parameters
        self.fusion_model['weights'] -= learning_rate * dw
        self.fusion_model['bias'] -= learning_rate * db
    
    def _update_linear_regression(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        """Update linear regression parameters."""
        weights = self.fusion_model['weights']
        bias = self.fusion_model['bias']
        
        # Forward pass
        predictions = np.dot(X, weights) + bias
        
        # Gradients
        dw = np.dot(X.T, (predictions - y)) / len(X)
        db = np.mean(predictions - y)
        
        # Update parameters
        self.fusion_model['weights'] -= learning_rate * dw
        self.fusion_model['bias'] -= learning_rate * db
    
    def _update_neural_network(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        """Update neural network parameters."""
        # This is a simplified implementation
        # In practice, you might want to use a proper deep learning framework
        logger.warning("Neural network training not fully implemented")
    
    def _calculate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate loss for given data."""
        if self.fusion_method == 'logistic_regression':
            return self._logistic_loss(X, y)
        elif self.fusion_method == 'linear_regression':
            return self._mse_loss(X, y)
        else:
            return 0.0
    
    def _logistic_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate logistic loss."""
        weights = self.fusion_model['weights']
        bias = self.fusion_model['bias']
        
        logits = np.dot(X, weights) + bias
        predictions = 1 / (1 + np.exp(-logits))
        
        # Binary cross-entropy loss
        loss = -np.mean(y * np.log(predictions + 1e-8) + (1 - y) * np.log(1 - predictions + 1e-8))
        return loss
    
    def _mse_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate mean squared error loss."""
        weights = self.fusion_model['weights']
        bias = self.fusion_model['bias']
        
        predictions = np.dot(X, weights) + bias
        loss = np.mean((predictions - y) ** 2)
        return loss
    
    def _calculate_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy for given data."""
        predictions = self._get_fusion_scores(X)
        predictions = np.array(predictions)
        
        # Convert to binary predictions
        binary_predictions = (predictions > 0.5).astype(int)
        binary_labels = (y > 0.5).astype(int)
        
        accuracy = np.mean(binary_predictions == binary_labels)
        return accuracy
    
    def save_model(self, model_path: Optional[str] = None) -> None:
        """
        Save the trained fusion model.
        
        Args:
            model_path: Path to save model
        """
        model_path = model_path or self.model_path
        if not model_path:
            logger.warning("No model path provided")
            return
        
        with RetrievalLogger("Saving fusion model", logger):
            model_data = {
                'fusion_method': self.fusion_method,
                'fusion_model': self.fusion_model,
                'feature_names': self.feature_names,
                'default_weights': self.default_weights
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Fusion model saved to {model_path}")
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the trained fusion model.
        
        Args:
            model_path: Path to load model from
        """
        model_path = model_path or self.model_path
        if not model_path or not os.path.exists(model_path):
            logger.warning("Model file not found")
            return
        
        with RetrievalLogger("Loading fusion model", logger):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.fusion_method = model_data.get('fusion_method', self.fusion_method)
                self.fusion_model = model_data.get('fusion_model', self.fusion_model)
                self.feature_names = model_data.get('feature_names', self.feature_names)
                self.default_weights = model_data.get('default_weights', self.default_weights)
                
                logger.info(f"Fusion model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load fusion model: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the fusion model.
        
        Returns:
            Model information
        """
        return {
            'fusion_method': self.fusion_method,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'enable_online_learning': self.enable_online_learning,
            'learning_rate': self.learning_rate
        }
