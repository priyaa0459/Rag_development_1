#!/bin/bash

# Load SAP iFlow data to vector database with OpenAI multi-model strategy
echo "Loading SAP iFlow data to vector database with OpenAI multi-model strategy..."

# Run the vector loader script
python ../src/vector_loader_openai.py

echo "Data loading complete!"
