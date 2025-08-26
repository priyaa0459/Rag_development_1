# embedding_models.py
from sentence_transformers import SentenceTransformer

def load_sentence_transformer_model(model_name: str):
    """Load a SentenceTransformer model."""
    try:
        model = SentenceTransformer(model_name)
        print(f"Loaded model: {model_name}")
        return model
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return None
