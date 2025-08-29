"""
SAP iFlow RAG Generation Pipeline (Person 3)
Person 3: RAG Generation using Cohere AI

This module implements:
- Context building from Person 2 SearchResult objects
- Cohere AI-powered code generation for SAP iFlow
- Specialized prompt templates for different output types
- Code validation and formatting
"""

import os
import json
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import cohere
try:
    from src.retriever import SearchResult
except ImportError:
    from retriever import SearchResult

# Load environment variables
load_dotenv()

@dataclass
class GenerationRequest:
    """Structured representation of a generation request."""
    query: str
    search_results: List[SearchResult]
    output_type: str  # 'xml', 'groovy', 'properties', 'integration_steps'
    max_tokens: int = 2048
    temperature: float = 0.3

@dataclass
class GeneratedOutput:
    """Structured representation of generated output."""
    query: str
    output_type: str
    generated_code: str
    validation_status: str
    confidence_score: float
    context_chunks_used: List[str]
    generation_metadata: Dict[str, Any]

class SAPiFlowGenerator:
    """
    RAG generation system for SAP iFlow using Cohere AI.
    Generates code and configuration from retrieved search results.
    """
    
    def __init__(self):
        """Initialize the generator with Cohere API."""
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        if not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY must be set in .env file")
        
        self.co = cohere.Client(self.cohere_api_key)
        self.models_initialized = True
        print("✓ Cohere AI client initialized")
        
        # Prompt templates for different output types
        self.prompt_templates = self._initialize_prompt_templates()
    
    def _initialize_prompt_templates(self) -> Dict[str, str]:
        """Initialize specialized prompt templates for SAP iFlow generation."""
        return {
            'xml': """You are an expert SAP Cloud Integration developer. Generate complete, valid SAP iFlow XML configuration based on the user query and context.

User Query: {query}

Context from SAP iFlow dataset:
{context}

Instructions:
1. Generate complete, valid SAP iFlow XML configuration
2. Include all necessary elements: iflow, integration, sender, receiver, mapping, etc.
3. Use proper XML syntax and SAP Cloud Integration schema
4. Make the configuration production-ready and well-structured
5. Include relevant parameters and error handling

Generate the complete SAP iFlow XML configuration:""",

            'groovy': """You are an expert SAP Cloud Integration developer. Generate complete, valid Groovy scripts for SAP iFlow message mapping based on the user query and context.

User Query: {query}

Context from SAP iFlow dataset:
{context}

Instructions:
1. Generate complete, valid Groovy script for message mapping
2. Include proper imports and error handling
3. Use SAP Cloud Integration specific APIs and patterns
4. Make the script production-ready with logging
5. Include comments explaining the logic

Generate the complete Groovy script:""",

            'properties': """You are an expert SAP Cloud Integration developer. Generate complete, valid property configurations for SAP iFlow based on the user query and context.

User Query: {query}

Context from SAP iFlow dataset:
{context}

Instructions:
1. Generate complete property configuration
2. Include all necessary parameters and values
3. Use proper SAP Cloud Integration property format
4. Make the configuration production-ready
5. Include comments for each property

Generate the complete property configuration:""",

            'integration_steps': """You are an expert SAP Cloud Integration developer. Generate detailed integration steps and configuration for SAP iFlow based on the user query and context.

User Query: {query}

Context from SAP iFlow dataset:
{context}

Instructions:
1. Generate step-by-step integration configuration
2. Include all necessary components and settings
3. Use proper SAP Cloud Integration terminology
4. Make the configuration production-ready
5. Include error handling and monitoring steps

Generate the complete integration steps and configuration:"""
        }
    
    def _dedup_search_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Filter duplicate contexts by instruction, chunk_id, or output_code while preserving order."""
        if not results:
            return results
        seen_instr = set()
        seen_chunks = set()
        seen_code = set()
        unique: List[SearchResult] = []
        for r in results:
            instr = (r.instruction or "").strip()
            cid = (r.chunk_id or "").strip()
            code = (r.output_code or "").strip()
            if instr and instr in seen_instr:
                continue
            if cid and cid in seen_chunks:
                continue
            if code and code in seen_code:
                continue
            unique.append(r)
            if instr:
                seen_instr.add(instr)
            if cid:
                seen_chunks.add(cid)
            if code:
                seen_code.add(code)
        return unique
    
    def build_context_from_results(self, search_results: List[SearchResult]) -> str:
        """
        Build context string from SearchResult objects for prompt generation.
        
        Args:
            search_results: List of SearchResult objects from Person 2
            
        Returns:
            Formatted context string for prompts
        """
        if not search_results:
            return "No relevant context found."
        
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            # Build context for each result
            result_context = f"Result {i}:\n"
            result_context += f"  Type: {result.output_type}\n"
            result_context += f"  Instruction: {result.instruction}\n"
            result_context += f"  Input Context: {result.input_context}\n"
            result_context += f"  Output Code: {result.output_code}\n"
            result_context += f"  Similarity Score: {result.codebert_similarity:.3f}\n"
            
            if result.metadata:
                meta = result.metadata
                if 'data_type' in meta:
                    result_context += f"  Data Type: {meta['data_type']}\n"
                if 'source_file' in meta:
                    result_context += f"  Source: {meta['source_file']}\n"
            
            context_parts.append(result_context)
        
        return "\n".join(context_parts)
    
    def generate_code(self, request: GenerationRequest) -> GeneratedOutput:
        """
        Generate SAP iFlow code using Cohere AI based on search results.
        
        Args:
            request: GenerationRequest object with query and context
            
        Returns:
            GeneratedOutput object with generated code and metadata
        """
        print(f"\n🚀 Generating {request.output_type} code for query: '{request.query}'")
        print("=" * 60)
        
        try:
            # Deduplicate search results to ensure diverse context
            original_count = len(request.search_results)
            unique_results = self._dedup_search_results(request.search_results)
            if len(unique_results) < original_count:
                print(f"⚠️ Duplicate contexts filtered: {original_count - len(unique_results)} removed. Using {len(unique_results)} unique chunks.")
            if len(unique_results) < max(1, min(3, original_count)):
                print(f"⚠️ Low diversity context: only {len(unique_results)} unique chunks available for prompt.")

            # Build context from unique results
            context = self.build_context_from_results(unique_results)
            print(f"📚 Built context from {len(unique_results)} search results")
            
            # Get appropriate prompt template
            if request.output_type not in self.prompt_templates:
                raise ValueError(f"Unsupported output type: {request.output_type}")
            
            prompt_template = self.prompt_templates[request.output_type]
            
            # Build the complete prompt
            prompt = prompt_template.format(
                query=request.query,
                context=context
            )
            
            print("🤖 Sending request to Cohere AI...")
            
            # Generate using Cohere
            response = self.co.generate(
                model='command',  # Use Cohere's latest command model
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                k=0,
                stop_sequences=[],
                return_likelihoods='NONE'
            )
            
            generated_text = response.generations[0].text.strip()
            
            print("✅ Code generation completed")
            
            # Validate the generated code
            validation_status = self._validate_generated_code(generated_text, request.output_type)
            
            # Calculate confidence score (simplified - could be enhanced) on unique results
            confidence_score = self._calculate_confidence_score(unique_results)
            
            # Create output object
            output = GeneratedOutput(
                query=request.query,
                output_type=request.output_type,
                generated_code=generated_text,
                validation_status=validation_status,
                confidence_score=confidence_score,
                context_chunks_used=[f"{r.chunk_id}: {r.instruction[:100]}..." for r in unique_results],
                generation_metadata={
                    'model': 'cohere-command',
                    'max_tokens': request.max_tokens,
                    'temperature': request.temperature,
                    'prompt_length': len(prompt),
                    'generation_id': response.generations[0].id if response.generations else None
                }
            )
            
            # Display the generated output
            self._display_generated_output(output)
            
            return output
            
        except Exception as e:
            print(f"❌ Error in code generation: {e}")
            # Return error output
            return GeneratedOutput(
                query=request.query,
                output_type=request.output_type,
                generated_code=f"Error generating code: {str(e)}",
                validation_status="ERROR",
                confidence_score=0.0,
                context_chunks_used=[],
                generation_metadata={'error': str(e)}
            )
    
    def _validate_generated_code(self, code: str, output_type: str) -> str:
        """
        Validate generated code for syntax and structure.
        
        Args:
            code: Generated code string
            output_type: Type of output to validate
            
        Returns:
            Validation status string
        """
        try:
            if output_type == 'xml':
                # Basic XML validation
                ET.fromstring(code)
                return "VALID_XML"
            elif output_type == 'groovy':
                # Basic Groovy syntax check (simplified)
                if 'import' in code and ('def ' in code or 'class ' in code):
                    return "VALID_GROOVY"
                else:
                    return "POTENTIALLY_INVALID_GROOVY"
            elif output_type == 'properties':
                # Properties format check
                if '=' in code and len(code.strip()) > 0:
                    return "VALID_PROPERTIES"
                else:
                    return "POTENTIALLY_INVALID_PROPERTIES"
            else:
                return "UNKNOWN_TYPE"
        except ET.ParseError:
            return "INVALID_XML"
        except Exception:
            return "VALIDATION_ERROR"
    
    def _calculate_confidence_score(self, search_results: List[SearchResult]) -> float:
        """
        Calculate confidence score based on search result quality.
        
        Args:
            search_results: List of SearchResult objects
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not search_results:
            return 0.0
        
        # Calculate average similarity score
        avg_similarity = sum(r.codebert_similarity for r in search_results) / len(search_results)
        
        # Boost confidence if we have multiple high-quality results
        if len(search_results) >= 3:
            avg_similarity *= 1.1
        
        # Cap at 1.0
        return min(avg_similarity, 1.0)
    
    def _display_generated_output(self, output: GeneratedOutput):
        """Display the generated output in a formatted way."""
        print(f"\n📊 Generated {output.output_type.upper()} Output:")
        print("-" * 60)
        print(f"🔍 Query: {output.query}")
        print(f"📝 Output Type: {output.output_type}")
        print(f"✅ Validation: {output.validation_status}")
        print(f"🎯 Confidence: {output.confidence_score:.3f}")
        print(f"📚 Context Chunks Used: {len(output.context_chunks_used)}")
        
        print(f"\n💻 Generated Code:")
        print("=" * 60)
        print(output.generated_code)
        print("=" * 60)
        
        print(f"\n📋 Generation Metadata:")
        for key, value in output.generation_metadata.items():
            print(f"   {key}: {value}")
        
        print("\n" + "=" * 60)
        print("💡 Generated output is ready for Person 4 (UI/Testing)")
        print("🔗 Use the GeneratedOutput object for downstream processing")
    
    def generate_from_query(self, query: str, search_results: List[SearchResult], 
                           output_type: str = 'xml') -> GeneratedOutput:
        """
        Convenience method to generate code directly from query and search results.
        
        Args:
            query: User query string
            search_results: List of SearchResult objects from Person 2
            output_type: Type of output to generate
            
        Returns:
            GeneratedOutput object
        """
        request = GenerationRequest(
            query=query,
            search_results=search_results,
            output_type=output_type
        )
        
        return self.generate_code(request)


def demo_generation():
    """Demo function to showcase the generation system."""
    print("🚀 SAP iFlow RAG Generation System Demo")
    print("=" * 50)
    
    # Check if Cohere API key is available
    if not os.getenv("COHERE_API_KEY"):
        print("❌ Error: COHERE_API_KEY must be set in .env file")
        print("Please add your Cohere API key to the .env file")
        return
    
    try:
        # Initialize generator
        generator = SAPiFlowGenerator()
        
        # Demo queries and expected output types
        demo_requests = [
            {
                'query': "Create an invoice request integration flow",
                'output_type': 'xml',
                'description': 'SAP iFlow XML configuration'
            },
            {
                'query': "Groovy script for message mapping",
                'output_type': 'groovy',
                'description': 'Groovy message mapping script'
            },
            {
                'query': "HTTP adapter properties configuration",
                'output_type': 'properties',
                'description': 'HTTP adapter properties'
            }
        ]
        
        print(f"\n🧪 Running {len(demo_requests)} demo generations...")
        
        for i, demo_req in enumerate(demo_requests, 1):
            print(f"\n{'='*20} Demo Generation {i} {'='*20}")
            print(f"Query: {demo_req['query']}")
            print(f"Expected Output: {demo_req['description']}")
            
            # Note: In a real scenario, you would get search_results from Person 2
            # For demo purposes, we'll create mock results
            mock_results = [
                SearchResult(
                    chunk_id="demo_chunk_1",
                    instruction=demo_req['query'],
                    input_context="Demo input context for testing",
                    output_code="Demo output code",
                    output_type="DEMO",
                    embedding_model="demo",
                    codebert_similarity=0.9,
                    cross_encoder_score=0.8,
                    hybrid_score=0.85,
                    metadata={'data_type': 'demo', 'source_file': 'demo.json'}
                )
            ]
            
            # Generate code
            output = generator.generate_from_query(
                demo_req['query'],
                mock_results,
                demo_req['output_type']
            )
            
            if i < len(demo_requests):
                print("\n⏳ Waiting 3 seconds before next generation...")
                import time
                time.sleep(3)
        
        print(f"\n✅ Demo complete! Generation system is ready for Person 4 integration.")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")


def interactive_generation():
    """Interactive generation interface for testing."""
    print("🔍 Interactive SAP iFlow Generation")
    print("Type 'quit' to exit, 'help' for usage info")
    
    # Check if Cohere API key is available
    if not os.getenv("COHERE_API_KEY"):
        print("❌ Error: COHERE_API_KEY must be set in .env file")
        return
    
    try:
        # Initialize generator
        generator = SAPiFlowGenerator()
        
        while True:
            try:
                query = input("\n🔍 Enter your query: ").strip()
                
                if query.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif query.lower() == 'help':
                    print("Available output types: xml, groovy, properties, integration_steps")
                    print("Example: 'Create SOAP adapter configuration' -> xml")
                    continue
                elif not query:
                    continue
                
                # Get output type
                output_type = input("📝 Enter output type (xml/groovy/properties/integration_steps): ").strip().lower()
                if output_type not in ['xml', 'groovy', 'properties', 'integration_steps']:
                    output_type = 'xml'  # Default to XML
                
                # Note: In a real scenario, you would get search_results from Person 2
                # For demo purposes, we'll create mock results
                mock_results = [
                    SearchResult(
                        chunk_id="interactive_chunk_1",
                        instruction=query,
                        input_context="Interactive input context",
                        output_code="Interactive output code",
                        output_type="INTERACTIVE",
                        embedding_model="interactive",
                        codebert_similarity=0.85,
                        cross_encoder_score=0.8,
                        hybrid_score=0.825,
                        metadata={'data_type': 'interactive', 'source_file': 'interactive.json'}
                    )
                ]
                
                # Generate code
                output = generator.generate_from_query(query, mock_results, output_type)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Error in interactive generation: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_generation()
        elif sys.argv[1] == "interactive":
            interactive_generation()
        else:
            print("Usage: python generator.py [demo|interactive]")
            print("  demo: Run demonstration generations")
            print("  interactive: Start interactive generation interface")
    else:
        # Default: run demo
        demo_generation()
