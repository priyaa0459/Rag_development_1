"""
SAP iFlow Embedding Model Research Script (Updated for OpenAI Multi-Model Strategy)
Person 1: Vector Storage and Embedding Research

This script evaluates different embedding models for SAP iFlow code generation,
saves JSON results and a Markdown report into the results/ directory.

Models tested:
- OpenAI text-embedding-ada-002 (1536D)
- Microsoft CodeBERT (768D → padded to 1536D)
- Sentence Transformers MiniLM (384D → padded to 1536D)
"""

import os
from dotenv import load_dotenv
import json
import pandas as pd
import numpy as np
import time
import sys
from typing import List, Dict, Tuple

# Try to import required packages
try:
    from sentence_transformers import SentenceTransformer
    print("✓ Sentence Transformers imported successfully")
except ImportError:
    print("✗ Sentence Transformers not available. Install with: pip install sentence-transformers")
    sys.exit(1)

try:
    from openai import OpenAI
    print("✓ OpenAI imported successfully")
except ImportError:
    print("✗ OpenAI not available. Install with: pip install openai")
    sys.exit(1)

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import silhouette_score
    print("✓ Scikit-learn imported successfully")
except ImportError:
    print("✗ Scikit-learn not available. Install with: pip install scikit-learn")
    sys.exit(1)

# Paths
PROCESSED_CSV = os.path.join("data", "processed", "processed_sap_iflow_data.csv")
RESULTS_JSON = os.path.join("results", "embedding_evaluation_results_openai_multimodel.json")
REPORT_MD = os.path.join("results", "embedding_research_report_openai_multimodel.md")


class EmbeddingModelEvaluator:
    """Evaluate different embedding models for SAP iFlow code with multi-model strategy."""

    def __init__(self):
        self.models = {
            "text-embedding-ada-002": None,
            "microsoft/codebert-base": None,
            "sentence-transformers/all-MiniLM-L6-v2": None,
        }
        self.results = {}
        self.openai_client = None
        self.target_dimension = 1536
        
    def initialize_models(self):
        """Initialize all embedding models."""
        print("Initializing multi-model embedding system...")
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                self.openai_client = OpenAI(api_key=api_key)
                print("✓ OpenAI client initialized")
            except Exception as e:
                print(f"✗ OpenAI initialization failed: {e}")
                self.openai_client = None
        
        # Initialize CodeBERT
        try:
            self.models["microsoft/codebert-base"] = SentenceTransformer("microsoft/codebert-base")
            print("✓ CodeBERT model loaded (768D)")
        except Exception as e:
            print(f"✗ Failed to load CodeBERT: {e}")

        # Initialize MiniLM
        try:
            self.models["sentence-transformers/all-MiniLM-L6-v2"] = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            print("✓ MiniLM model loaded (384D)")
        except Exception as e:
            print(f"✗ Failed to load MiniLM: {e}")

    def pad_embedding_to_target_dimension(self, embedding: np.ndarray, source_dim: int) -> np.ndarray:
        """Pad or truncate embedding to target 1536 dimensions."""
        if source_dim == self.target_dimension:
            return embedding
        elif source_dim < self.target_dimension:
            # Pad with zeros
            padded = np.pad(embedding, (0, self.target_dimension - source_dim), 'constant')
            return padded
        else:
            # Truncate
            return embedding[:self.target_dimension]

    def generate_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Generate embeddings for given texts using specified model."""
        if model_name == "text-embedding-ada-002":
            return self._generate_openai_embeddings(texts, model_name)
        else:
            return self._generate_local_embeddings(texts, model_name)

    def _generate_openai_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Generate OpenAI embeddings in batches."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        batch_size = 100  # OpenAI supports up to 100 texts per request
        all_embs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = self.openai_client.embeddings.create(
                    input=batch, model=model_name
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embs.extend(batch_embeddings)
                print(f"Generated OpenAI embeddings for {i + len(batch)}/{len(texts)} texts")
                time.sleep(0.1)  # Small delay to be respectful to API
            except Exception as e:
                print(f"Error generating OpenAI embeddings for batch {i}: {e}")
                # Add zero vectors as fallback
                zero_embedding = [0.0] * self.target_dimension
                all_embs.extend([zero_embedding] * len(batch))

        return np.array(all_embs)

    def _generate_local_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Generate local model embeddings and pad to target dimension."""
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not loaded")
        
        # Generate embeddings
        embeddings = model.encode(texts, convert_to_numpy=True)
        source_dim = embeddings.shape[1]
        
        # Pad to target dimension
        if source_dim != self.target_dimension:
            print(f"Padding {model_name} embeddings from {source_dim}D to {self.target_dimension}D")
            padded_embeddings = []
            for emb in embeddings:
                padded = self.pad_embedding_to_target_dimension(emb, source_dim)
                padded_embeddings.append(padded)
            embeddings = np.array(padded_embeddings)
        
        return embeddings

    def evaluate_model_performance(self, df: pd.DataFrame, sample_size: int = 50) -> Dict:
        """Evaluate embedding model performance on SAP iFlow data."""
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        results = {}

        for name in self.models:
            if name == "text-embedding-ada-002" and not self.openai_client:
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
                    "was_padded": name != "text-embedding-ada-002",
                    "original_dimension": 768 if "codebert" in name else (384 if "minilm" in name else 1536)
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
        
        # Add multi-model strategy recommendations
        recs.append("\n**Multi-Model Strategy Recommendations**:")
        recs.append("- **Primary**: Use OpenAI text-embedding-ada-002 for general queries")
        recs.append("- **Code Queries**: Use CodeBERT for queries containing code-related keywords")
        recs.append("- **Fallback**: Use MiniLM when OpenAI API is unavailable")
        recs.append("- **Storage**: All embeddings padded to 1536D for consistency")
        
        return "\n".join(recs)


def main():
    # Load environment variables from .env file
    load_dotenv()
    print("SAP iFlow Embedding Model Research (OpenAI Multi-Model Strategy)")
    print("=" * 80)

    if not os.path.exists(PROCESSED_CSV):
        print(f"Error: processed CSV not found at {PROCESSED_CSV}")
        sys.exit(1)

    df = pd.read_csv(PROCESSED_CSV)
    print(f"Loaded {len(df)} samples from {PROCESSED_CSV}")

    evaluator = EmbeddingModelEvaluator()
    evaluator.initialize_models()
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
        "# SAP iFlow Embedding Model Evaluation Report (OpenAI Multi-Model Strategy)\n",
        "## Model Strategy\n",
        "- **Primary**: OpenAI text-embedding-ada-002 (1536D)\n",
        "- **Code Queries**: Microsoft CodeBERT (768D → padded to 1536D)\n",
        "- **Fallback**: Sentence Transformers MiniLM (384D → padded to 1536D)\n\n",
        "## Results\n",
        "```json\n"]
    
    md.append(json.dumps(results, indent=2, default=lambda o: o.item() if hasattr(o, "item") else o))
    md.append("```\n## Recommendations\n")
    md.append(recs)
    md.append("\n\n## Model Comparison\n")
    md.append("| Model | Dim | Time/Text | Memory(MB) | Clust.Score | Padded |\n")
    md.append("|-------|-----|-----------|------------|-------------|--------|\n")
    for name, r in results.items():
        if "error" in r:
            continue
        cs = f"{r.get('clustering_score', 'N/A'):.3f}" if "clustering_score" in r else "N/A"
        padded = "Yes" if r.get('was_padded', False) else "No"
        md.append(
            f"| {name} | {r['embedding_dimension']} | {r['avg_time_per_text']:.3f}s | {r['memory_usage_mb']:.1f} | {cs} | {padded} |\n"
        )
    md.append("\n\n## Technical Details\n")
    md.append("- **Target Dimension**: 1536D for consistent storage\n")
    md.append("- **Padding Strategy**: Zero-padding for smaller embeddings\n")
    md.append("- **Model Selection**: Intelligent routing based on query content\n")
    md.append("- **Fallback Logic**: Graceful degradation on API failures\n")
    md.append("\n\n*Generated on " + time.strftime("%Y-%m-%d %H:%M:%S") + "*\n")

    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"✔ Saved Markdown report to {REPORT_MD}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
