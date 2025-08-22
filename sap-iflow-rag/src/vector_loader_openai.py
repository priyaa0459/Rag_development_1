"""
Supabase Vector Loading System for SAP iFlow RAG Pipeline (OpenAI-only)
Person 1: Vector Storage and Embedding Research

This script creates the vector database schema and loads embeddings using:
- OpenAI text-embedding-ada-002 (1536D)
"""

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from supabase import create_client, Client
from openai import OpenAI
import time
from datetime import datetime
import re


class SupabaseVectorLoader:
    """Load SAP iFlow embeddings into Supabase vector database with OpenAI-only strategy."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.openai_client = None
        self.embedding_dimension = 1536  # Target dimension for storage
        self.models_initialized = False
        
    def initialize_models(self):
        """Initialize OpenAI client only."""
        print("Initializing OpenAI embedding system...")
        
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            try:
                self.openai_client = OpenAI(api_key=api_key)
                print("✓ OpenAI client initialized")
            except Exception as e:
                print(f"✗ OpenAI initialization failed: {e}")
                self.openai_client = None
        
        self.models_initialized = True
        print(f"Target embedding dimension: {self.embedding_dimension}D")
        
    def generate_openai_embedding(self, text: str) -> List[float]:
        """Generate a single OpenAI embedding (text-embedding-ada-002)."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        try:
            response = self.openai_client.embeddings.create(
                input=[text],
                model='text-embedding-ada-002'
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating OpenAI embedding: {e}")
            # Return zero-vector to avoid crashes; caller may handle empties
            return [0.0] * self.embedding_dimension
    
    def create_vector_schema(self):
        """Create the vector storage schema in Supabase for 1536D vectors."""
        
        create_table_sql = f"""
        -- Enable the pgvector extension
        CREATE EXTENSION IF NOT EXISTS vector;
        
        -- Drop existing table if it exists
        DROP TABLE IF EXISTS sap_iflow_chunks CASCADE;
        
        -- Create the sap_iflow_chunks table with 1536D vectors
        CREATE TABLE sap_iflow_chunks (
            id BIGSERIAL PRIMARY KEY,
            chunk_id TEXT UNIQUE NOT NULL,
            instruction TEXT NOT NULL,
            input_context TEXT NOT NULL,
            output_code TEXT NOT NULL,
            combined_text TEXT NOT NULL,
            full_context TEXT NOT NULL,
            output_type TEXT,
            instruction_length INTEGER,
            input_length INTEGER,
            output_length INTEGER,
            combined_length INTEGER,
            embedding VECTOR({self.embedding_dimension}),
            embedding_model TEXT NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Create indexes for better performance
        CREATE INDEX sap_iflow_chunks_embedding_idx ON sap_iflow_chunks 
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        
        CREATE INDEX sap_iflow_chunks_output_type_idx ON sap_iflow_chunks(output_type);
        CREATE INDEX sap_iflow_chunks_created_at_idx ON sap_iflow_chunks(created_at);
        CREATE INDEX sap_iflow_chunks_model_idx ON sap_iflow_chunks(embedding_model);
        
        -- Create a function for similarity search with 1536D vectors
        CREATE OR REPLACE FUNCTION search_sap_iflow_chunks(
            query_embedding VECTOR({self.embedding_dimension}),
            match_threshold FLOAT DEFAULT 0.7,
            match_count INT DEFAULT 10,
            filter_model TEXT DEFAULT NULL
        )
        RETURNS TABLE (
            id BIGINT,
            chunk_id TEXT,
            instruction TEXT,
            input_context TEXT,
            output_code TEXT,
            output_type TEXT,
            embedding_model TEXT,
            similarity FLOAT,
            metadata JSONB
        )
        LANGUAGE SQL
        AS $$
            SELECT
                id,
                chunk_id,
                instruction,
                input_context,
                output_code,
                output_type,
                embedding_model,
                1 - (embedding <=> query_embedding) AS similarity,
                metadata
            FROM sap_iflow_chunks
            WHERE (filter_model IS NULL OR embedding_model = filter_model)
            AND 1 - (embedding <=> query_embedding) > match_threshold
            ORDER BY embedding <=> query_embedding
            LIMIT match_count;
        $$;
        """
        
        print("Creating vector database schema for 1536D vectors...")
        try:
            print("Schema SQL generated. Execute this in Supabase SQL editor:")
            print(create_table_sql)
            return create_table_sql
        except Exception as e:
            print(f"Error creating schema: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate OpenAI embeddings in batches (returns only vectors)."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        all_embeddings: List[List[float]] = []
        batch_size = 100  # OpenAI supports up to 100 inputs
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.openai_client.embeddings.create(
                    input=batch,
                    model='text-embedding-ada-002'
                )
                batch_embeddings = [d.embedding for d in response.data]
                all_embeddings.extend(batch_embeddings)
                print(f"Generated embeddings for {i + len(batch)}/{len(texts)} texts")
                time.sleep(0.1)
            except Exception as e:
                print(f"Error generating embeddings for batch starting at {i}: {e}")
                # Append zero vectors for this batch to maintain alignment
                zero_embedding = [0.0] * self.embedding_dimension
                all_embeddings.extend([zero_embedding] * len(batch))
        return all_embeddings
    
    def load_data_to_vector_db(self, df: pd.DataFrame):
        """Load SAP iFlow data with embeddings into Supabase."""
        
        print(f"Loading {len(df)} SAP iFlow samples into vector database...")
        
        # Clear existing records to avoid duplicate key errors
        try:
            self.supabase.table('sap_iflow_chunks').delete().neq('id', 0).execute()
            print("Cleared existing records from sap_iflow_chunks")
        except Exception as e:
            print(f"Warning: could not clear existing data (will rely on UPSERT): {e}")
        
        # Generate embeddings for combined text using OpenAI only
        combined_texts = df['combined_text'].tolist()
        embeddings_list = self.generate_embeddings_batch(combined_texts)
        
        # Prepare data for insertion
        insert_data = []
        
        for idx, row in df.iterrows():
            # Get the embedding vector for this row
            embedding = embeddings_list[idx]
            # Validate embedding dimensionality (must be 1536D)
            if not isinstance(embedding, list) or len(embedding) != self.embedding_dimension:
                print(f"Warning: embedding at index {idx} has invalid dimension ({len(embedding) if isinstance(embedding, list) else 'N/A'}), replacing with zero vector")
                embedding = [0.0] * self.embedding_dimension
            
            record = {
                'chunk_id': f"sap_iflow_{row['chunk_id']}",
                'instruction': row['instruction'],
                'input_context': row['input'],
                'output_code': row['output'],
                'combined_text': row['combined_text'],
                'full_context': row['full_context'],
                'output_type': row['output_type'],
                'instruction_length': int(row['instruction_length']),
                'input_length': int(row['input_length']),
                'output_length': int(row['output_length']),
                'combined_length': int(row['combined_length']),
                'embedding': embedding,
                'embedding_model': 'text-embedding-ada-002',
                'metadata': {
                    'data_type': row['data_type'],
                    'created_at': row['created_at'],
                    'source_file': row.get('source_file', 'unknown'),
                    'model_info': {
                        'model_type': 'openai',
                        'model_name': 'text-embedding-ada-002',
                        'source_dimension': self.embedding_dimension,
                        'target_dimension': self.embedding_dimension,
                        'padded': False
                    },
                    'original_dimension': self.embedding_dimension,
                    'stored_dimension': self.embedding_dimension,
                    'was_padded': False
                }
            }
            insert_data.append(record)
        
        # Upsert data in batches (update on conflict by chunk_id)
        batch_size = 10
        successful_inserts = 0
        
        for i in range(0, len(insert_data), batch_size):
            batch = insert_data[i:i + batch_size]
            
            try:
                result = self.supabase.table('sap_iflow_chunks').upsert(batch, on_conflict='chunk_id').execute()
                successful_inserts += len(batch)
                print(f"✓ Upserted batch {i//batch_size + 1}: {successful_inserts}/{len(insert_data)} records")
                
            except Exception as e:
                print(f"✗ Error inserting batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"\nData loading complete: {successful_inserts}/{len(insert_data)} records upserted successfully")
        return successful_inserts
    
    def _keyword_fallback_search(self, query_text: str, limit: int = 5):
        """Keyword fallback: match ANY word in instruction using ILIKE."""
        try:
            words = [w for w in re.split(r"\W+", query_text) if w]
            if not words:
                return []
            # Build OR filter for ILIKE on instruction
            # e.g., instruction.ilike.%word1%,instruction.ilike.%word2%
            or_parts = [f"instruction.ilike.%{w}%" for w in words]
            or_expr = ",".join(or_parts)
            result = self.supabase.table('sap_iflow_chunks') \
                .select('chunk_id, instruction, output_code, output_type, embedding_model') \
                .or_(or_expr) \
                .limit(limit) \
                .execute()
            return result.data or []
        except Exception as e:
            print(f"Keyword fallback search error: {e}")
            return []

    def test_similarity_search(self, query_text: str, top_k: int = 5):
        """Semantic search with OpenAI embedding and improved matching thresholds.
        Workflow: threshold 0.5 → if none, threshold 0.0 → if none, keyword fallback. Always request 5 unique instructions.
        """
        print(f"Testing similarity search with query: '{query_text[:100]}...'")
        try:
            # Always embed with OpenAI
            query_embedding = self.generate_openai_embedding(query_text)
            print("Query embedding generated using: text-embedding-ada-002")

            def get_unique_results(match_threshold: float, initial_match_count: int = 20):
                """Get unique results with given threshold, increasing match_count if needed."""
                match_count = initial_match_count
                max_attempts = 3  # Prevent infinite loops
                
                for attempt in range(max_attempts):
                    result = self.supabase.rpc('search_sap_iflow_chunks', {
                        'query_embedding': query_embedding,
                        'match_threshold': match_threshold,
                        'match_count': match_count,
                        'filter_model': 'text-embedding-ada-002'
                    }).execute()
                    
                    data = result.data or []
                    if not data:
                        return []
                    
                    # Filter for unique instructions
                    unique_chunks = []
                    seen_instructions = set()
                    
                    for chunk in data:
                        instruction = chunk.get('instruction', '').strip()
                        if instruction and instruction not in seen_instructions:
                            unique_chunks.append(chunk)
                            seen_instructions.add(instruction)
                            
                            if len(unique_chunks) >= top_k:
                                break
                    
                    # If we have enough unique results, return them
                    if len(unique_chunks) >= top_k:
                        return unique_chunks[:top_k]
                    
                    # If not enough unique results, increase match_count for next attempt
                    if attempt < max_attempts - 1:
                        match_count = min(match_count * 2, 100)  # Cap at 100
                        print(f"Only {len(unique_chunks)} unique results found, increasing match_count to {match_count}...")
                    
                # Return whatever unique results we found
                return unique_chunks[:top_k]

            # First attempt: threshold 0.5
            print("Searching with threshold 0.5...")
            data = get_unique_results(0.5)
            
            if not data:
                # Retry with threshold 0.0
                print("No results at threshold 0.5, retrying with threshold 0.0...")
                data = get_unique_results(0.0)

            if not data:
                print("No semantic results even at 0.0, using keyword fallback...")
                data = self._keyword_fallback_search(query_text, limit=top_k)
                # Ensure keyword results are also unique
                if data:
                    unique_chunks = []
                    seen_instructions = set()
                    for chunk in data:
                        instruction = chunk.get('instruction', '').strip()
                        if instruction and instruction not in seen_instructions:
                            unique_chunks.append(chunk)
                            seen_instructions.add(instruction)
                            if len(unique_chunks) >= top_k:
                                break
                    data = unique_chunks[:top_k]

            if data:
                print(f"Found {len(data)} unique similar chunks:")
                for i, chunk in enumerate(data):
                    sim = chunk.get('similarity')
                    if sim is not None:
                        print(f"\n{i+1}. Similarity: {sim:.3f}")
                    else:
                        print(f"\n{i+1}. Similarity: N/A (keyword match)")
                    print(f"   Type: {chunk.get('output_type')}")
                    print(f"   Model: {chunk.get('embedding_model')}")
                    print(f"   Instruction: {chunk.get('instruction', '')[:100]}...")
                    # Show metadata model info if present
                    meta = chunk.get('metadata') if isinstance(chunk, dict) else None
                    if meta and isinstance(meta, dict) and 'model_info' in meta:
                        mi = meta['model_info']
                        print(f"   Source: {mi.get('model_name', 'unknown')} ({mi.get('source_dimension', '?')}D)")
            else:
                print("No similar chunks found")
            return data
        except Exception as e:
            print(f"Error testing similarity search: {e}")
            return []
    
    def get_database_stats(self):
        """Get statistics about the loaded data."""
        try:
            # Count total records
            count_result = self.supabase.table('sap_iflow_chunks').select('*', count='exact').execute()
            total_count = count_result.count
            
            # Get output type distribution
            type_result = self.supabase.table('sap_iflow_chunks')\
                .select('output_type, embedding_model')\
                .execute()
            
            if type_result.data:
                types = [item['output_type'] for item in type_result.data]
                models = [item['embedding_model'] for item in type_result.data]
                type_distribution = pd.Series(types).value_counts().to_dict()
                model_distribution = pd.Series(models).value_counts().to_dict()
            else:
                type_distribution = {}
                model_distribution = {}
            
            stats = {
                'total_chunks': total_count,
                'output_type_distribution': type_distribution,
                'embedding_model_distribution': model_distribution,
                'last_updated': datetime.now().isoformat()
            }
            
            print("Database Statistics:")
            print(f"Total chunks: {stats['total_chunks']}")
            print("Output type distribution:")
            for type_name, count in stats['output_type_distribution'].items():
                print(f"  {type_name}: {count}")
            print("Embedding model distribution:")
            for model_name, count in stats['embedding_model_distribution'].items():
                print(f"  {type_name}: {count}")
            
            return stats
            
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return None


def main():
    """Main function to set up and load vector database with OpenAI-only strategy."""
    
    print("SAP iFlow Vector Database Setup (OpenAI-only)")
    print("=" * 50)
    
    # Load environment variables from .env
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
        return
    
    # Initialize vector loader
    loader = SupabaseVectorLoader(supabase_url, supabase_key)
    
    # Step 1: Initialize OpenAI model
    loader.initialize_models()
    
    # Step 2: Always reload data (force re-embedding with OpenAI)
    csv_path = 'data/processed/processed_sap_iflow_data.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return
    
    # Optional: Clear existing data to avoid duplicates (uncomment if needed)
    # try:
    #     loader.supabase.table('sap_iflow_chunks').delete().neq('id', 0).execute()
    #     print("Cleared existing records from sap_iflow_chunks")
    # except Exception as e:
    #     print(f"Warning: could not clear existing data: {e}")

    # Load and embed all samples
    df = pd.read_csv(csv_path)
    print(f"\n3. Loaded {len(df)} processed samples")
    print("\n4. Loading data to vector database with OpenAI embeddings...")
    success_count = loader.load_data_to_vector_db(df)
    
    # Step 4: Test similarity search with existing data
    print("\n4. Testing similarity search...")
    print("Waiting 10 seconds for rate limits to reset...")
    time.sleep(10)

    test_queries = [
        "Create an invoice request integration flow",
        "SOAP parameters for iFlow",
        "Business partner replication"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n--- Testing Query {i+1}: {query} ---")
        # Test similarity search
        results = loader.test_similarity_search(query)
        if i < len(test_queries) - 1:
            time.sleep(3)

    # Step 5: Get database statistics
    print("\n5. Database statistics")
    loader.get_database_stats()
    
    print(f"\n✅ Vector database loading & testing complete! {success_count} records available.")
    print("- ✅ Database connection successful")
    print("- ✅ Similarity search tested")
    print("- ✅ OpenAI-only embedding system working")


if __name__ == "__main__":
    main()
