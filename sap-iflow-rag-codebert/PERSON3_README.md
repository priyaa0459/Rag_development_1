# Person 3: SAP iFlow RAG Generation Pipeline

## Overview
Person 3 implements the RAG generation pipeline using Cohere AI models to generate SAP iFlow code from retrieved search results.

## Features
- **Cohere AI Integration**: Uses Cohere's latest `command` model for code generation
- **Multiple Output Types**: XML, Groovy, Properties, Integration Steps
- **Context-Aware Generation**: Builds prompts from Person 2 SearchResult objects
- **Code Validation**: Basic syntax validation for generated code
- **Confidence Scoring**: Based on search result quality

## Setup
1. Add `COHERE_API_KEY` to your `.env` file
2. Install dependencies: `pip install -r requirements.txt`

## Usage

### Basic Generation
```python
from src.generator import SAPiFlowGenerator
from src.retriever import SearchResult

# Initialize generator
generator = SAPiFlowGenerator()

# Generate code from search results
output = generator.generate_from_query(
    query="Create invoice request integration flow",
    search_results=search_results,  # From Person 2
    output_type='xml'
)
```

### Supported Output Types
- `xml`: SAP iFlow XML configuration
- `groovy`: Groovy message mapping scripts
- `properties`: Property configurations
- `integration_steps`: Integration step configurations

## Commands

### Demo Mode
```bash
python src/generator.py demo
```

### Interactive Mode
```bash
python src/generator.py interactive
```

### Complete Pipeline Test
```bash
python test_pipeline.py
```

## Integration with Person 2
Person 3 seamlessly integrates with Person 2's SearchResult objects:

```python
# Person 2: Retrieve results
retriever = SAPiFlowRetriever(supabase_url, supabase_key)
search_results = retriever.search("your query")

# Person 3: Generate code
generator = SAPiFlowGenerator()
output = generator.generate_from_query("your query", search_results, 'xml')
```

## Output Structure
Generated output includes:
- Generated code/configuration
- Validation status
- Confidence score
- Context chunks used
- Generation metadata

## Ready for Person 4
The system provides structured output ready for UI integration and testing.

