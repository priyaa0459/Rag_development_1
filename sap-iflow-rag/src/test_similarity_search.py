"""
Simple similarity search test for existing SAP iFlow data
"""

from dotenv import load_dotenv
load_dotenv()

import os
from supabase import create_client

def test_similarity_search():
    """Test similarity search on existing data."""
    
    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
        return
    
    supabase = create_client(supabase_url, supabase_key)
    
    # Test queries
    test_queries = [
        "Create an invoice request integration flow",
        "SOAP parameters for iFlow", 
        "Business partner replication"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n{'='*60}")
        print(f"Query {i+1}: {query}")
        print(f"{'='*60}")
        
        try:
            # Search by instruction text
            result = supabase.table('sap_iflow_chunks') \
                .select('chunk_id, instruction, output_code, output_type, embedding_model') \
                .or_(f'instruction.ilike.%{query}%,output_code.ilike.%{query}%') \
                .limit(5) \
                .execute()
            
            if result.data:
                print(f"Found {len(result.data)} similar chunks:")
                for j, chunk in enumerate(result.data, 1):
                    print(f"\n{j}. Type: {chunk['output_type']}")
                    print(f"   Model: {chunk['embedding_model']}")
                    print(f"   Instruction: {chunk['instruction'][:100]}...")
                    print(f"   Output: {chunk['output_code'][:80]}...")
            else:
                # Try broader search
                print("No direct matches, trying broader search...")
                words = query.split()
                broader_result = supabase.table('sap_iflow_chunks') \
                    .select('chunk_id, instruction, output_code, output_type, embedding_model') \
                    .or_(f'instruction.ilike.%{words[0]}%,output_code.ilike.%{words[0]}%') \
                    .limit(5) \
                    .execute()
                
                if broader_result.data:
                    print(f"Found {len(broader_result.data)} broader matches:")
                    for j, chunk in enumerate(broader_result.data, 1):
                        print(f"\n{j}. Type: {chunk['output_type']}")
                        print(f"   Model: {chunk['embedding_model']}")
                        print(f"   Instruction: {chunk['instruction'][:100]}...")
                        print(f"   Output: {chunk['output_code'][:80]}...")
                else:
                    print("No similar chunks found")
                    
        except Exception as e:
            print(f"Error searching: {e}")

if __name__ == "__main__":
    test_similarity_search()
