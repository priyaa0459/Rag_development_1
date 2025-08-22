"""
SAP iFlow Data Loader
Loads and prepares SAP iFlow data for processing
"""

import pandas as pd
import os

def load_sap_iflow_data(csv_path: str = None) -> pd.DataFrame:
    """Load SAP iFlow data from CSV file."""
    
    if csv_path is None:
        csv_path = 'data/processed/processed_sap_iflow_data.csv'
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading SAP iFlow data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"✓ Loaded {len(df)} samples")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Output types: {df['output_type'].unique()}")
    
    return df

def get_data_info(df: pd.DataFrame) -> dict:
    """Get information about the loaded data."""
    
    info = {
        'total_samples': len(df),
        'output_types': df['output_type'].value_counts().to_dict(),
        'data_types': df['data_type'].value_counts().to_dict() if 'data_type' in df.columns else {},
        'avg_instruction_length': df['instruction_length'].mean() if 'instruction_length' in df.columns else 0,
        'avg_output_length': df['output_length'].mean() if 'output_length' in df.columns else 0
    }
    
    return info

if __name__ == "__main__":
    try:
        # Load data
        df = load_sap_iflow_data()
        
        # Get info
        info = get_data_info(df)
        
        print("\nData Summary:")
        print(f"Total samples: {info['total_samples']}")
        print(f"Output types: {info['output_types']}")
        print(f"Data types: {info['data_types']}")
        print(f"Avg instruction length: {info['avg_instruction_length']:.1f}")
        print(f"Avg output length: {info['avg_output_length']:.1f}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
