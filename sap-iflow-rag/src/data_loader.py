# src/data_loader.py

"""
Data Loader for SAP iFlow Training Datasets

Supports:
- HF JSON: { "data": [ {...}, ... ] }
- Alpaca JSON: { "samples": [ {...}, ... ] }
- Zipped lists: { "instruction": [...], "input": [...], "output": [...] }
- JSON Lines (each line is a JSON object)
"""

import json
import pandas as pd
from datetime import datetime
import os
import sys

def classify_output_type(output: str) -> str:
    o = output.lower()
    if '<?xml' in o and 'bpmn2' in o:       return 'BPMN2_XML'
    if '<?xml' in o and 'project' in o:     return 'PROJECT_XML'
    if '<?xml' in o and 'wsdl' in o:        return 'WSDL'
    if '<?xml' in o:                        return 'GENERIC_XML'
    if 'def ' in o and 'groovy' in o:       return 'GROOVY_SCRIPT'
    if 'parameter' in o and 'key' in o:     return 'PARAMETER_CONFIG'
    if 'properties' in o:                   return 'PROPERTIES_CONFIG'
    if o.startswith('openapi'):             return 'OPENAPI_SPEC'
    return 'OTHER'

def load_json_file(file_path: str) -> list:
    """Load a JSON or JSONL file and extract samples."""
    samples = []
    # JSONL: each line is a JSON object
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        return samples

    with open(file_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # HF format
    if isinstance(raw, dict):
        if 'data' in raw and isinstance(raw['data'], list):
            return raw['data']
        if 'samples' in raw and isinstance(raw['samples'], list):
            return raw['samples']
        instr = raw.get('instruction'); inp = raw.get('input'); out = raw.get('output')
        if isinstance(instr, list) and isinstance(inp, list) and isinstance(out, list):
            n = min(len(instr), len(inp), len(out))
            return [{'instruction': instr[i], 'input': inp[i], 'output': out[i]} for i in range(n)]
    # fallback empty
    return []

def load_and_preprocess_multiple(json_files: list, csv_output_path: str):
    """Load multiple input files, combine samples, preprocess, and save CSV."""
    all_samples = []

    for file_path in json_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        samples = load_json_file(file_path)
        print(f"Loaded {len(samples)} samples from {file_path}")
        for s in samples:
            s['source_file'] = os.path.basename(file_path)
        all_samples.extend(samples)

    if not all_samples:
        raise RuntimeError("No samples loaded from any files.")

    print(f"Total combined samples: {len(all_samples)}")
    df = pd.DataFrame(all_samples)

    # Features
    df['combined_text'] = df['instruction'] + ' ' + df['input']
    df['full_context']  = df['combined_text'] + ' ' + df['output']
    df['instruction_length'] = df['instruction'].str.len()
    df['input_length']       = df['input'].str.len()
    df['output_length']      = df['output'].str.len()
    df['combined_length']    = df['combined_text'].str.len()
    df['output_type']        = df['output'].apply(classify_output_type)

    # Metadata
    df['chunk_id']   = range(1, len(df) + 1)
    df['created_at'] = datetime.utcnow().isoformat()
    df['data_type']  = 'sap_iflow_training'

    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    df.to_csv(csv_output_path, index=False)
    print(f"✔ Saved {len(df)} samples to {csv_output_path}")

    print("\nBreakdown by source:")
    for fname, cnt in df['source_file'].value_counts().items():
        print(f"  {fname}: {cnt}")

if __name__ == '__main__':
    # Default files
    default_inputs = [
        'data/raw/sap_iflow_hf_dataset_20250818_210539.json',
        'data/raw/sap_iflow_alpaca_20250818_210539.json',
        'data/raw/sap_iflow_unsloth_20250818_210539.json',
        'data/raw/sap_iflow_training_20250818_210539.jsonl'
    ]
    default_output = 'data/processed/processed_sap_iflow_data.csv'

    args = sys.argv[1:]
    if len(args) == 0:
        inputs = default_inputs
        output = default_output
    elif len(args) == 1:
        inputs = default_inputs
        output = args[0]
    else:
        inputs = args[:-1]
        output = args[-1]

    load_and_preprocess_multiple(inputs, output)
