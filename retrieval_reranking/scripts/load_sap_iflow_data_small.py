#!/usr/bin/env python3
"""
Load a subset of SAP iFlow dataset for faster testing.
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
from retrieval.retrieval_pipeline import RetrievalPipeline
from utils.logging_utils import get_logger, RetrievalLogger

load_dotenv()
logger = get_logger(__name__)


def load_sap_iflow_data_subset(max_documents=50):
    """Load a subset of SAP iFlow data for faster testing."""
    
    data_dir = Path("data")
    documents = []
    
    # Load from processed CSV (subset)
    csv_path = data_dir / "processed" / "processed_sap_iflow_data.csv"
    if csv_path.exists():
        logger.info(f"Loading subset of processed data from {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            # Take only first max_documents
            df_subset = df.head(max_documents)
            logger.info(f"Loading {len(df_subset)} records from CSV (subset)")
            
            for idx, row in df_subset.iterrows():
                # Create document from CSV row
                doc = {
                    'id': f"csv_{idx}",
                    'text': row.get('combined_text', ''),
                    'instruction': row.get('instruction', ''),
                    'input': row.get('input', ''),
                    'output': row.get('output', ''),
                    'source_file': row.get('source_file', ''),
                    'output_type': row.get('output_type', ''),
                    'chunk_id': row.get('chunk_id', ''),
                    'metadata': {
                        'component_type': _classify_component_type(row.get('instruction', '')),
                        'complexity': _calculate_complexity(row.get('instruction', '')),
                        'reusability': _calculate_reusability(row.get('instruction', '')),
                        'freshness': 1.0,
                        'popularity': 0.5,
                        'data_type': row.get('data_type', 'processed'),
                        'instruction_length': row.get('instruction_length', 0),
                        'input_length': row.get('input_length', 0),
                        'output_length': row.get('output_length', 0)
                    }
                }
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
    
    # If CSV doesn't have enough data, load from JSONL
    if len(documents) < max_documents:
        jsonl_path = data_dir / "raw" / "sap_iflow_training_20250818_210539.jsonl"
        if jsonl_path.exists():
            logger.info(f"Loading subset from {jsonl_path}")
            try:
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        if idx >= max_documents - len(documents):
                            break
                        if line.strip():
                            data = json.loads(line)
                            doc = {
                                'id': f"jsonl_{idx}",
                                'text': f"{data.get('instruction', '')} {data.get('input', '')}",
                                'instruction': data.get('instruction', ''),
                                'input': data.get('input', ''),
                                'output': data.get('output', ''),
                                'metadata': {
                                    'component_type': _classify_component_type(data.get('instruction', '')),
                                    'complexity': _calculate_complexity(data.get('instruction', '')),
                                    'reusability': _calculate_reusability(data.get('instruction', '')),
                                    'freshness': 1.0,
                                    'popularity': 0.5,
                                    'data_type': 'raw_jsonl'
                                }
                            }
                            documents.append(doc)
                            
                logger.info(f"Added {len([d for d in documents if d['id'].startswith('jsonl_')])} records from JSONL")
                            
            except Exception as e:
                logger.error(f"Error loading JSONL: {e}")
    
    logger.info(f"Total documents loaded (subset): {len(documents)}")
    return documents


def _classify_component_type(instruction: str) -> str:
    """Simple rule-based component type classification."""
    instruction_lower = instruction.lower()
    
    if any(word in instruction_lower for word in ['trigger', 'webhook', 'event', 'start', 'initiate']):
        return 'trigger'
    elif any(word in instruction_lower for word in ['transform', 'convert', 'map', 'format', 'parse']):
        return 'transformer'
    elif any(word in instruction_lower for word in ['connect', 'integration', 'api', 'service']):
        return 'connector'
    elif any(word in instruction_lower for word in ['error', 'exception', 'retry', 'fallback']):
        return 'error_handler'
    elif any(word in instruction_lower for word in ['condition', 'if', 'else', 'switch', 'validate']):
        return 'condition'
    elif any(word in instruction_lower for word in ['aggregate', 'sum', 'count', 'batch', 'collect']):
        return 'aggregator'
    elif any(word in instruction_lower for word in ['map', 'mapping', 'field', 'schema']):
        return 'data_mapper'
    else:
        return 'action'


def _calculate_complexity(instruction: str) -> str:
    """Calculate complexity based on instruction length and keywords."""
    length = len(instruction)
    complex_keywords = ['complex', 'advanced', 'multiple', 'batch', 'aggregate', 'transform']
    
    if length > 500 or any(keyword in instruction.lower() for keyword in complex_keywords):
        return 'high'
    elif length > 200:
        return 'medium'
    else:
        return 'low'


def _calculate_reusability(instruction: str) -> float:
    """Calculate reusability score based on instruction content."""
    instruction_lower = instruction.lower()
    
    reusable_keywords = ['generic', 'template', 'reusable', 'common', 'standard', 'configurable']
    specific_keywords = ['specific', 'custom', 'unique', 'particular', 'dedicated']
    
    reusable_count = sum(1 for keyword in reusable_keywords if keyword in instruction_lower)
    specific_count = sum(1 for keyword in specific_keywords if keyword in instruction_lower)
    
    score = 0.5
    score += reusable_count * 0.1
    score -= specific_count * 0.1
    
    return max(0.0, min(1.0, score))


def main():
    """Main function to load SAP iFlow data subset and test the system."""
    
    print("🚀 Loading SAP iFlow Dataset (Small Subset)")
    print("=" * 50)
    
    # Load documents (subset)
    documents = load_sap_iflow_data_subset(max_documents=50)
    
    if not documents:
        print("❌ No documents loaded. Please check your data files.")
        return
    
    print(f"✅ Loaded {len(documents)} SAP iFlow documents (subset)")
    
    # Initialize pipeline
    print("\n🔧 Initializing Retrieval Pipeline...")
    pipeline = RetrievalPipeline()
    
    # Add documents to pipeline
    print(f"\n📚 Adding {len(documents)} documents to pipeline...")
    with RetrievalLogger("Adding SAP iFlow documents subset", logger):
        pipeline.add_documents(documents)
    
    print("✅ Documents added successfully!")
    
    # Test with SAP-specific queries
    print("\n🧪 Testing with SAP iFlow Queries")
    print("=" * 50)
    
    test_queries = [
        "Create SAP Cloud Integration project structure for invoice processing",
        "Configure HTTP adapter for S4HANA integration",
        "Transform JSON to XML for SAP message format",
        "Handle errors in SAP integration flow",
        "Map fields between supplier and buyer systems"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        
        try:
            results = pipeline.search(query, top_k=3)
            
            if results and 'results' in results:
                for j, result in enumerate(results['results'][:3], 1):
                    doc = result['document']
                    score = result['score']
                    component_type = doc.get('metadata', {}).get('component_type', 'unknown')
                    
                    print(f"  {j}. Score: {score:.3f} | Type: {component_type}")
                    print(f"     Instruction: {doc.get('instruction', '')[:100]}...")
                    
            if 'query_classification' in results and results['query_classification']:
                classification = results['query_classification']
                print(f"  Query Type: {classification.get('primary_type', 'unknown')}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n🎉 SAP iFlow dataset subset integration complete!")
    print(f"📊 Total documents indexed: {len(documents)}")
    print(f"🔍 System ready for SAP iFlow queries!")
    print(f"💡 This is a subset. Run the full version for complete dataset.")


if __name__ == "__main__":
    main()
