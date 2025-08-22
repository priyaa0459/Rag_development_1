# save_results.py
import json
import os
from datetime import datetime

def save_json(data, filepath):
    """Save data as JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved JSON to {filepath}")

def save_markdown(content, filepath):
    """Save content as Markdown file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved Markdown to {filepath}")

# Example usage
if __name__ == "__main__":
    example_results = {"test": "data", "timestamp": datetime.now()}
    md_report = "# Test Report\n\nThis is a test markdown report."
    
    save_json(example_results, 'results/embedding_evaluation_results_openai_multimodel.json')
    save_markdown(md_report, 'results/embedding_research_report_openai_multimodel.md')
