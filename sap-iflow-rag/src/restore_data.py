"""
Restore SAP iFlow data to database with proper dimensions
"""

from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
from supabase import create_client

def restore_data():
    """Restore data to the database."""
    
    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
        return
    
    supabase = create_client(supabase_url, supabase_key)
    
    # Load CSV data
    csv_path = 'data/processed/processed_sap_iflow_data.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from CSV")
    
    # Check if table exists and has data
    try:
        count_result = supabase.table('sap_iflow_chunks').select('*', count='exact').execute()
        existing_count = count_result.count
        print(f"Database currently has {existing_count} records")
        
        if existing_count > 0:
            print("Database already has data. Skipping restore.")
            return
            
    except Exception as e:
        print(f"Table doesn't exist or error: {e}")
        print("Creating table structure...")
        
        # Create simple table structure
        create_sql = """
        -- Enable the pgvector extension
        CREATE EXTENSION IF NOT EXISTS vector;
        
        -- Create the sap_iflow_chunks table
        CREATE TABLE IF NOT EXISTS sap_iflow_chunks (
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
            embedding_model TEXT DEFAULT 'text-embedding-ada-002',
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS sap_iflow_chunks_output_type_idx ON sap_iflow_chunks(output_type);
        CREATE INDEX IF NOT EXISTS sap_iflow_chunks_created_at_idx ON sap_iflow_chunks(created_at);
        CREATE INDEX IF NOT EXISTS sap_iflow_chunks_model_idx ON sap_iflow_chunks(embedding_model);
        """
        
        print("Please execute this SQL in your Supabase SQL editor:")
        print(create_sql)
        print("\nAfter creating the table, press Enter to continue...")
        input()
    
    # Insert data without embeddings for now
    print("Inserting data without embeddings...")
    
    insert_data = []
    for idx, row in df.iterrows():
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
            'embedding_model': 'text-embedding-ada-002',
            'metadata': {
                'data_type': row['data_type'],
                'created_at': row['created_at'],
                'source_file': row.get('source_file', 'unknown')
            }
        }
        insert_data.append(record)
    
    # Insert in batches
    batch_size = 10
    successful_inserts = 0
    
    for i in range(0, len(insert_data), batch_size):
        batch = insert_data[i:i + batch_size]
        
        try:
            result = supabase.table('sap_iflow_chunks').insert(batch).execute()
            successful_inserts += len(batch)
            print(f"✓ Inserted batch {i//batch_size + 1}: {successful_inserts}/{len(insert_data)} records")
            
        except Exception as e:
            print(f"✗ Error inserting batch {i//batch_size + 1}: {e}")
            continue
    
    print(f"\nData restore complete: {successful_inserts}/{len(insert_data)} records inserted successfully")

if __name__ == "__main__":
    restore_data()


