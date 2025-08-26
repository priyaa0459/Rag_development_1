# src/save_results.py

import os
import json

def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"✔ JSON results saved to {path}")

def save_markdown(text: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"✔ Markdown report saved to {path}")

if __name__ == '__main__':
    # Example usage
    example_results = {'model': 'all-MiniLM-L6-v2', 'avg_time': 0.123, 'accuracy': 0.85}
    save_json(example_results, 'results/embedding_evaluation_results_cohere.json')

    md_report = "# Example Report\n\nThis is your detailed evaluation."
    save_markdown(md_report, 'results/embedding_research_report_cohere.md')
