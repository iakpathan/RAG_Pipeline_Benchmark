"""
RAG + HyDE Benchmark — Full Implementation
Run: python main.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from indexer import build_index, load_passages
from pipelines import standard_rag, hyde_rag, multi_query_rag
from evaluator import run_evaluation
from visualizer import plot_results

# ─────────────────────────────────────────────
# CONFIG — tweak these freely
# ─────────────────────────────────────────────
CORPUS_SIZE      = 2_000
NUM_EVAL_QUERIES = 20      # queries to evaluate (start small, scale up)
TOP_K            = 5        # documents retrieved per query
EMBED_MODEL      = "BAAI/bge-small-en-v1.5"
LLM_MODEL        = "llama-3.1-8b-instant"   # Groq model
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Set GROQ_API_KEY in your .env file. Get it free at groq.com")

# ─────────────────────────────────────────────
# STEP 1: Build index
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Loading corpus and building FAISS index")
print("="*60)

passages, queries, ground_truth = load_passages(CORPUS_SIZE, NUM_EVAL_QUERIES)
index, embed_model = build_index(passages, EMBED_MODEL)

print(f"  Indexed {len(passages):,} passages")
print(f"  Eval queries: {len(queries)}")

# ─────────────────────────────────────────────
# STEP 2: Define all three pipelines
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Setting up pipelines")
print("="*60)

pipelines = {
    "Standard RAG":    lambda q: standard_rag(q,   index, embed_model, passages, TOP_K),
    "HyDE RAG":        lambda q: hyde_rag(q,        index, embed_model, passages, TOP_K, GROQ_API_KEY, LLM_MODEL),
    "Multi-Query RAG": lambda q: multi_query_rag(q, index, embed_model, passages, TOP_K, GROQ_API_KEY, LLM_MODEL),
}
print(f"  Pipelines ready: {list(pipelines.keys())}")

# ─────────────────────────────────────────────
# STEP 3: Evaluate
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Running evaluation with RAGAS")
print("="*60)

all_results = run_evaluation(
    pipelines=pipelines,
    queries=queries,
    ground_truth=ground_truth,
    groq_api_key=GROQ_API_KEY,
    llm_model=LLM_MODEL,
)

# ─────────────────────────────────────────────
# STEP 4: Visualize
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Generating comparison dashboard")
print("="*60)

plot_results(all_results)
print("\n  Saved: results/comparison_dashboard.png")
print("  Saved: results/scores.json")
print("\nDone!")
