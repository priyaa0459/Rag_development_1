"""
SAP iFlow Embedding Model Research Script (Updated for Cohere)
Person 1: Vector Storage and Embedding Research

This script evaluates different embedding models for SAP iFlow code generation,
saves JSON results and a Markdown report into the results/ directory.
"""

import os
from dotenv import load_dotenv
import json
import pandas as pd
import numpy as np
import time
import sys
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import cohere
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score


# Paths
PROCESSED_CSV = os.path.join("data", "processed", "processed_sap_iflow_data.csv")
RESULTS_JSON = os.path.join("results", "embedding_evaluation_results_cohere.json")
REPORT_MD = os.path.join("results", "embedding_research_report_cohere.md")


class EmbeddingModelEvaluator:
    """Evaluate different embedding models for SAP iFlow code."""

    def __init__(self):
        self.models = {
            "all-MiniLM-L6-v2": None,
            "all-mpnet-base-v2": None,
            "codebert-base": None,
            "cohere-embed-english-v3": None,
            "cohere-embed-multilingual-v3": None,
        }
        self.results = {}
        self.cohere_client = None

    def initialize_cohere(self):
        """Initialize Cohere client."""
        api_key = os.getenv("COHERE_API_KEY")
        if api_key:
            self.cohere_client = cohere.Client(api_key)
            print("✓ Cohere client initialized")
        else:
            print("✗ Cohere API key not found in environment")

    def load_models(self):
        """Load embedding models."""
        print("Loading embedding models...")
        self.initialize_cohere()

        try:
            self.models["all-MiniLM-L6-v2"] = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            print("✓ Loaded all-MiniLM-L6-v2")
        except Exception as e:
            print(f"✗ Failed to load all-MiniLM-L6-v2: {e}")

        try:
            self.models["all-mpnet-base-v2"] = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2"
            )
            print("✓ Loaded all-mpnet-base-v2")
        except Exception as e:
            print(f"✗ Failed to load all-mpnet-base-v2: {e}")

        try:
            self.models["codebert-base"] = SentenceTransformer("microsoft/codebert-base")
            print("✓ Loaded codebert-base")
        except Exception as e:
            print(f"✗ Failed to load codebert-base: {e}")

    def generate_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Generate embeddings for given texts using specified model."""
        if model_name.startswith("cohere-embed"):
            return self._generate_cohere_embeddings(texts, model_name)
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not loaded")
        return model.encode(texts, convert_to_numpy=True)

    def _generate_cohere_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Generate Cohere embeddings in batches."""
        if not self.cohere_client:
            raise ValueError("Cohere client not initialized")

        model_map = {
            "cohere-embed-english-v3": "embed-english-v3.0",
            "cohere-embed-multilingual-v3": "embed-multilingual-v3.0",
        }
        cohere_model = model_map.get(model_name, "embed-english-v3.0")
        batch_size = 96
        all_embs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.cohere_client.embed(
                texts=batch, model=cohere_model, input_type="search_document"
            )
            all_embs.extend(response.embeddings)
            print(f"Generated Cohere embeddings for {i + len(batch)}/{len(texts)} texts")
            time.sleep(0.1)

        return np.array(all_embs)

    def evaluate_model_performance(self, df: pd.DataFrame, sample_size: int = 50) -> Dict:
        """Evaluate embedding model performance on SAP iFlow data."""
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        results = {}

        for name in self.models:
            if name.startswith("cohere-embed") and not self.cohere_client:
                print(f"Skipping {name} (no API key)")
                continue

            try:
                print(f"\nEvaluating {name}...")
                start = time.time()
                embs = self.generate_embeddings(sample_df["combined_text"].tolist(), name)
                duration = time.time() - start

                dims = embs.shape[1]
                mem = embs.nbytes / 1024 / 1024
                sim = cosine_similarity(embs)
                avg_sim = np.mean(sim[np.triu_indices_from(sim, k=1)])
                std_sim = np.std(sim[np.triu_indices_from(sim, k=1)])

                res = {
                    "model_name": name,
                    "embedding_time": duration,
                    "avg_time_per_text": duration / len(sample_df),
                    "embedding_dimension": dims,
                    "sample_size": len(sample_df),
                    "memory_usage_mb": mem,
                    "avg_similarity": avg_sim,
                    "similarity_std": std_sim,
                }

                if len(sample_df["output_type"].unique()) > 1:
                    res["clustering_score"] = self._evaluate_clustering(
                        embs, sample_df["output_type"]
                    )

                results[name] = res
                print(f"✓ {name}: {dims}D, {res['avg_time_per_text']:.3f}s/text, {mem:.2f}MB")
            except Exception as e:
                print(f"✗ Error evaluating {name}: {e}")
                results[name] = {"error": str(e)}

        return results

    def _evaluate_clustering(self, embs: np.ndarray, labels: pd.Series) -> float:
        """Compute silhouette score for clustering by output type."""
        label_map = {l: i for i, l in enumerate(labels.unique())}
        nums = labels.map(label_map)
        try:
            return silhouette_score(embs, nums)
        except Exception as e:
            print(f"Clustering evaluation failed: {e}")
            return 0.0

    def generate_recommendations(self, results: Dict) -> str:
        """Create human-readable recommendations based on metrics."""
        valid = {k: v for k, v in results.items() if "error" not in v}
        if not valid:
            return "No valid model results."

        fastest = min(valid, key=lambda k: valid[k]["avg_time_per_text"])
        efficient = min(valid, key=lambda k: valid[k]["memory_usage_mb"])
        clustering = {k: v for k, v in valid.items() if "clustering_score" in v}
        best_cluster = max(clustering, key=lambda k: clustering[k]["clustering_score"]) if clustering else None

        recs = [
            f"**Fastest model**: {fastest} ({valid[fastest]['avg_time_per_text']:.3f}s/text)",
            f"**Most memory efficient**: {efficient} ({valid[efficient]['memory_usage_mb']:.2f}MB)",
        ]
        if best_cluster:
            score = clustering[best_cluster]["clustering_score"]
            recs.append(f"**Best clustering**: {best_cluster} (score {score:.3f})")
        recs.append(
            "\n**Recommendation**: Choose based on your priorities—speed, memory, or clustering quality."
        )
        return "\n".join(recs)


def main():
    # Load environment variables from .env file
    load_dotenv()
    print("SAP iFlow Embedding Model Research (with Cohere)")
    print("=" * 60)

    if not os.path.exists(PROCESSED_CSV):
        print(f"Error: processed CSV not found at {PROCESSED_CSV}")
        sys.exit(1)

    df = pd.read_csv(PROCESSED_CSV)
    print(f"Loaded {len(df)} samples from {PROCESSED_CSV}")

    evaluator = EmbeddingModelEvaluator()
    evaluator.load_models()
    results = evaluator.evaluate_model_performance(df, sample_size=30)

    # Save JSON results with numpy type conversion for serialization
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(
            results, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else o
        )
    print(f"✔ Saved JSON results to {RESULTS_JSON}")

    # Build Markdown report with safe serialization
    recs = evaluator.generate_recommendations(results)
    md = [
        "# SAP iFlow Embedding Model Evaluation Report (with Cohere)\n",
        "## Results\n",
        "```\n"]
    
    md.append(json.dumps(results, indent=2, default=lambda o: o.item() if hasattr(o, "item") else o))
    md.append("```\n## Recommendations\n")
    md.append(recs)
    md.append("\n\n## Model Comparison\n")
    md.append("| Model | Dim | Time/Text | Memory(MB) | Clust.Score |\n")
    md.append("|-------|-----|-----------|------------|-------------|\n")
    for name, r in results.items():
        if "error" in r:
            continue
        cs = f"{r.get('clustering_score', 'N/A'):.3f}" if "clustering_score" in r else "N/A"
        md.append(
            f"| {name} | {r['embedding_dimension']} | {r['avg_time_per_text']:.3f}s | {r['memory_usage_mb']:.1f} | {cs} |\n"
        )
    md.append("\n\n*Generated on " + time.strftime("%Y-%m-%d %H:%M:%S") + "*\n")

    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"✔ Saved Markdown report to {REPORT_MD}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
