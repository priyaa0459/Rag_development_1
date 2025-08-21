#!/bin/bash

# Activate virtual environment (modify path if needed)
source sap-iflow-rag/Scripts/activate

echo "Starting vector data loading to Supabase..."
python ../src/vector_loader_cohere.py
