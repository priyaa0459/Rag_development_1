"""
Test SOAP Query with OpenAI Multi-Model Strategy
Tests different query types to demonstrate intelligent model selection
"""

from dotenv import load_dotenv
load_dotenv()

import os
from supabase import create_client
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np

def test_multi_model_queries():
    """Test different query types with intelligent model selection."""
    
    # Initialize clients
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))
    
    # Initialize OpenAI client
    openai_client = None
    try:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("✓ OpenAI client initialized")
    except Exception as e:
        print(f"✗ OpenAI initialization failed: {e}")
    
    # Initialize CodeBERT for code queries
    codebert_model = None
    try:
        codebert_model = SentenceTransformer('microsoft/codebert-base')
        print("✓ CodeBERT model loaded")
    except Exception as e:
        print(f"✗ CodeBERT initialization failed: {e}")
    
    # Initialize MiniLM as fallback
    minilm_model = None
    try:
        minilm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✓ MiniLM model loaded")
    except Exception as e:
        print(f"✗ MiniLM initialization failed: {e}")
    
    # Test queries with different characteristics
    test_queries = [
        {
            "text": "Configure SOAP adapter",
            "type": "general",
            "expected_model": "OpenAI"
        },
        {
            "text": "Create a Groovy script function for message processing",
            "type": "code",
            "expected_model": "CodeBERT"
        },
        {
            "text": "Set up API endpoint parameters",
            "type": "code",
            "expected_model": "CodeBERT"
        },
        {
            "text": "Business partner replication flow",
            "type": "general",
            "expected_model": "OpenAI"
        }
    ]
    
    for i, query_info in enumerate(test_queries):
        print(f"\n{'='*60}")
        print(f"Query {i+1}: {query_info['text']}")
        print(f"Type: {query_info['type']} | Expected Model: {query_info['expected_model']}")
        print(f"{'='*60}")
        
        try:
            # Generate embedding with intelligent model selection
            query_embedding, model_name, model_info = generate_embedding_intelligent(
                query_info['text'], 
                openai_client, 
                codebert_model, 
                minilm_model
            )
            
            print(f"Embedding generated using: {model_name}")
            print(f"Model info: {model_info}")
            
            # Test similarity search
            result = supabase.rpc('search_sap_iflow_chunks', {
                'query_embedding': query_embedding,
                'match_threshold': 0.0,  # Accept any similarity for testing
                'match_count': 5
            }).execute()
            
            if result.data:
                print(f"✅ Found {len(result.data)} results:")
                for j, match in enumerate(result.data):
                    print(f"\n{j+1}. Similarity: {match['similarity']:.3f}")
                    print(f"   Type: {match['output_type']}")
                    print(f"   Model: {match['embedding_model']}")
                    print(f"   Instruction: {match['instruction'][:80]}...")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"Error: {e}")

def generate_embedding_intelligent(query_text, openai_client, codebert_model, minilm_model):
    """Generate embedding with intelligent model selection."""
    
    query_lower = query_text.lower()
    
    # Check for code-specific keywords
    code_keywords = ['code', 'script', 'function', 'method', 'class', 'variable', 'parameter', 'api', 'endpoint']
    is_code_query = any(keyword in query_lower for keyword in code_keywords)
    
    # Try OpenAI first (if available)
    if openai_client:
        try:
            response = openai_client.embeddings.create(
                input=[query_text],
                model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding
            model_info = {
                'model_type': 'openai',
                'model_name': 'text-embedding-ada-002',
                'source_dimension': len(embedding),
                'target_dimension': 1536,
                'padded': False
            }
            return embedding, 'text-embedding-ada-002', model_info
        except Exception as e:
            print(f"OpenAI failed: {e}")
    
    # Try CodeBERT for code queries
    if is_code_query and codebert_model:
        try:
            embedding = codebert_model.encode([query_text], convert_to_numpy=True)[0]
            # Pad to 1536D
            if len(embedding) < 1536:
                padded = np.pad(embedding, (0, 1536 - len(embedding)), 'constant')
            else:
                padded = embedding[:1536]
            
            model_info = {
                'model_type': 'codebert',
                'model_name': 'microsoft/codebert-base',
                'source_dimension': len(embedding),
                'target_dimension': 1536,
                'padded': True
            }
            return padded.tolist(), 'microsoft/codebert-base', model_info
        except Exception as e:
            print(f"CodeBERT failed: {e}")
    
    # Try MiniLM as fallback
    if minilm_model:
        try:
            embedding = minilm_model.encode([query_text], convert_to_numpy=True)[0]
            # Pad to 1536D
            if len(embedding) < 1536:
                padded = np.pad(embedding, (0, 1536 - len(embedding)), 'constant')
            else:
                padded = embedding[:1536]
            
            model_info = {
                'model_type': 'minilm',
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'source_dimension': len(embedding),
                'target_dimension': 1536,
                'padded': True
            }
            return padded.tolist(), 'sentence-transformers/all-MiniLM-L6-v2', model_info
        except Exception as e:
            print(f"MiniLM failed: {e}")
    
    # If all else fails, return zero vector
    zero_embedding = [0.0] * 1536
    model_info = {
        'model_type': 'zero_fallback',
        'model_name': 'zero_vector',
        'source_dimension': 0,
        'target_dimension': 1536,
        'padded': False,
        'fallback_reason': 'all_models_failed'
    }
    return zero_embedding, 'zero_vector', model_info

if __name__ == "__main__":
    test_multi_model_queries()
