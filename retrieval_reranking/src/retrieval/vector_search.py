import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import faiss
import pickle
import os
from pathlib import Path

from utils.logging_utils import get_logger, RetrievalLogger
from utils.query_utils import normalize_query

# Optional imports: OpenAI and HF transformers
try:
    import openai  # type: ignore
except Exception:  # pragma: no cover
    openai = None  # type: ignore

try:
    import torch  # type: ignore
    from transformers import AutoTokenizer, AutoModel  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore

logger = get_logger(__name__)


class VectorSearch:
    """
    Vector search system using OpenAI embeddings and FAISS for efficient similarity search.
    """
    
    def __init__(self, openai_client: Optional[Any] = None, model_name: str = 'microsoft/codebert-base', index_path: Optional[str] = None, documents_path: Optional[str] = None, dimension: Optional[int] = None, backend: str = 'hf', device: Optional[str] = None):
        """
        Initialize the vector search system.
        
        Args:
            openai_client: OpenAI client instance (only if backend=='openai')
            model_name: Embedding model name ('microsoft/codebert-base' by default)
            index_path: Path to save/load FAISS index
            documents_path: Path to save/load document metadata
            dimension: Embedding dimension (auto-detected for HF if None)
            backend: 'hf' for HuggingFace models, 'openai' for OpenAI embeddings
            device: Computation device for HF models ('cuda'|'cpu'|None)
        """
        self.backend = backend
        self.model_name = model_name
        self.device = device or ("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
        self.dimension = dimension if dimension is not None else 768
        self.index_path = index_path
        self.documents_path = documents_path
        
        # Initialize embedding providers
        self.client = None
        self.tokenizer = None
        self.model = None
        if self.backend == 'openai':
            if openai is None:
                raise ImportError("openai package not available; install it or use backend='hf'")
            self.client = openai_client or openai.OpenAI()
            # default OpenAI embedding dimension if not provided
            if dimension is None:
                self.dimension = 1536
        else:
            if AutoTokenizer is None or AutoModel is None:
                raise ImportError("transformers/torch not available; install them or use backend='openai'")
            # Load HF model/tokenizer lazily here
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            # Infer hidden size if not provided
            try:
                hidden_size = getattr(self.model.config, 'hidden_size', None)
                if hidden_size and dimension is None:
                    self.dimension = int(hidden_size)
            except Exception:
                pass

        # Initialize FAISS index with resolved dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.documents = []
        self.document_embeddings = []
        
        # Load existing index if available
        if index_path and os.path.exists(index_path):
            self.load_index(index_path, documents_path)
        
        logger.info(f"Vector search initialized with model: {model_name} (backend={self.backend}, dim={self.dimension}, device={self.device})")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector search index.
        
        Args:
            documents: List of document dictionaries
        """
        with RetrievalLogger(f"Adding {len(documents)} documents to vector index", logger):
            for doc in documents:
                if 'text' not in doc:
                    logger.warning(f"Document missing 'text' field: {doc.get('id', 'unknown')}")
                    continue
                
                # Generate embedding
                embedding = self._get_embedding(doc['text'])
                if embedding is not None:
                    self.documents.append(doc)
                    self.document_embeddings.append(embedding)
            
            # Update FAISS index
            if self.document_embeddings:
                self._update_index()
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for text using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            if self.backend == 'openai':
                if self.client is None:
                    logger.error("OpenAI client not initialized")
                    return None
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=text
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            # HF backend: use CLS embedding from last hidden state
            if self.tokenizer is None or self.model is None or torch is None:
                logger.error("HF model/tokenizer not initialized")
                return None
            with torch.no_grad():
                encoded = self.tokenizer(text, truncation=True, max_length=512, padding=False, return_tensors='pt')
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                token_embeddings = outputs.last_hidden_state  # [1, seq_len, hidden]
                cls_embedding = token_embeddings[:, 0, :]  # [1, hidden]
                vec = cls_embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
                if vec.ndim != 1:
                    vec = vec.reshape(-1)
                # Enforce expected dimension by trimming/padding if necessary
                if vec.shape[0] > self.dimension:
                    vec = vec[:self.dimension]
                elif vec.shape[0] < self.dimension:
                    pad = np.zeros((self.dimension - vec.shape[0],), dtype=np.float32)
                    vec = np.concatenate([vec, pad], axis=0)
                return vec
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None
    
    def _update_index(self) -> None:
        """Update FAISS index with current embeddings."""
        if not self.document_embeddings:
            return
        
        # Filter embeddings that match expected dimension
        filtered_embeddings = [emb for emb in self.document_embeddings if isinstance(emb, np.ndarray) and emb.shape[0] == self.dimension]
        if len(filtered_embeddings) != len(self.document_embeddings):
            logger.warning(f"Filtered {len(self.document_embeddings) - len(filtered_embeddings)} embeddings with unexpected shape")
        if not filtered_embeddings:
            logger.warning("No valid embeddings to index")
            return
        # Convert to numpy array
        embeddings_array = np.stack(filtered_embeddings).astype(np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Clear and rebuild index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_array)
        
        logger.info(f"Updated FAISS index with {len(self.document_embeddings)} embeddings")
    
    def search(self, query: str, top_k: int = 10, threshold: float = 0.0, return_scores: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            return_scores: Whether to include similarity scores
            
        Returns:
            List of similar documents with scores
        """
        with RetrievalLogger(f"Vector search for query: {query[:50]}...", logger):
            # Normalize query
            normalized_query = normalize_query(query)
            
            # Get query embedding
            query_embedding = self._get_embedding(normalized_query)
            if query_embedding is None:
                logger.error("Failed to get query embedding")
                return []
            
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or score < threshold:  # FAISS returns -1 for invalid indices
                    continue
                
                result = {
                    'document': self.documents[idx].copy(),
                    'similarity_score': float(score),
                    'index': int(idx)
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} results above threshold {threshold}")
            return results
    
    def batch_search(self, queries: List[str], top_k: int = 10, threshold: float = 0.0) -> List[List[Dict[str, Any]]]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of top results per query
            threshold: Minimum similarity threshold
            
        Returns:
            List of result lists for each query
        """
        with RetrievalLogger(f"Batch vector search for {len(queries)} queries", logger):
            results = []
            for query in queries:
                query_results = self.search(query, top_k, threshold)
                results.append(query_results)
            return results
    
    def get_document_embedding(self, document_id: str) -> Optional[np.ndarray]:
        """
        Get embedding for a specific document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document embedding vector
        """
        for i, doc in enumerate(self.documents):
            if doc.get('id') == document_id:
                return self.document_embeddings[i]
        return None
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts using OpenAI embeddings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Get embeddings
            embedding1 = self._get_embedding(text1)
            embedding2 = self._get_embedding(text2)
            
            if embedding1 is None or embedding2 is None:
                return 0.0
            
            # Normalize embeddings
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def save_index(self, index_path: Optional[str] = None, documents_path: Optional[str] = None) -> None:
        """
        Save FAISS index and document metadata.
        
        Args:
            index_path: Path to save FAISS index
            documents_path: Path to save document metadata
        """
        index_path = index_path or self.index_path
        documents_path = documents_path or self.documents_path
        
        if not index_path or not documents_path:
            logger.warning("No save paths provided")
            return
        
        with RetrievalLogger("Saving vector search index", logger):
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save document metadata
            with open(documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            logger.info(f"Saved index to {index_path} and documents to {documents_path}")
    
    def load_index(self, index_path: Optional[str] = None, documents_path: Optional[str] = None) -> None:
        """
        Load FAISS index and document metadata.
        
        Args:
            index_path: Path to load FAISS index from
            documents_path: Path to load document metadata from
        """
        index_path = index_path or self.index_path
        documents_path = documents_path or self.documents_path
        
        if not index_path or not documents_path:
            logger.warning("No load paths provided")
            return
        
        if not os.path.exists(index_path) or not os.path.exists(documents_path):
            logger.warning("Index or documents file not found")
            return
        
        with RetrievalLogger("Loading vector search index", logger):
            try:
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load document metadata
                with open(documents_path, 'rb') as f:
                    self.documents = pickle.load(f)
                
                # Reconstruct embeddings from index (if needed)
                if hasattr(self.index, 'reconstruct'):
                    self.document_embeddings = []
                    for i in range(len(self.documents)):
                        embedding = self.index.reconstruct(i)
                        self.document_embeddings.append(embedding)
                
                logger.info(f"Loaded index with {len(self.documents)} documents")
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector search index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if hasattr(self.index, 'ntotal') else 0,
            'embedding_dimension': self.dimension,
            'model_name': self.model_name
        }
    
    def clear_index(self) -> None:
        """Clear all documents and embeddings from the index."""
        self.documents = []
        self.document_embeddings = []
        self.index = faiss.IndexFlatIP(self.dimension)
        logger.info("Cleared vector search index")
