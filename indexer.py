"""
indexer.py — Dataset loading and FAISS index construction
"""

import os
import pickle
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

CACHE_DIR = "cache"


def load_passages(corpus_size: int = 10_000, num_queries: int = 50):
    """
    Load MS MARCO passages and evaluation queries.
    MS MARCO structure:
      - ds['passages'] is a dict with 'passage_text' list and 'is_selected' list
      - ds['query'] is the question string
      - ds['answers'] is a list of gold answer strings
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = f"{CACHE_DIR}/passages_{corpus_size}_{num_queries}.pkl"

    if os.path.exists(cache_file):
        print("  Loading from cache...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("  Downloading MS MARCO v2.1 from HuggingFace...")
    ds = load_dataset(
        "ms_marco",
        "v2.1",
        split="train",
        trust_remote_code=True,
    )

    # ── Build passage corpus ──────────────────────────────────────────
    passages = []
    seen = set()
    for row in ds:
        for passage_text in row["passages"]["passage_text"]:
            text = passage_text.strip()
            if text and text not in seen and len(text) > 50:
                passages.append(text)
                seen.add(text)
            if len(passages) >= corpus_size:
                break
        if len(passages) >= corpus_size:
            break

    # ── Build eval queries + ground truth ────────────────────────────
    # Use rows that have at least one selected passage as "ground truth"
    queries = []
    ground_truth = []   # list of gold answer strings

    for row in ds:
        if len(queries) >= num_queries:
            break
        q = row["query"].strip()
        answers = row.get("answers", [])
        # Filter out empty / placeholder answers
        valid_answers = [a for a in answers if a and a != "No Answer Present."]
        if q and valid_answers:
            queries.append(q)
            ground_truth.append(valid_answers)

    print(f"  Passages: {len(passages):,}  |  Eval queries: {len(queries)}")

    data = (passages, queries, ground_truth)
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)

    return data


def build_index(passages: list, model_name: str = "BAAI/bge-small-en-v1.5"):
    """
    Encode passages with SentenceTransformer and build a FAISS IndexFlatIP
    (inner-product index — equivalent to cosine similarity after L2 normalisation).
    Returns (faiss_index, embed_model).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    index_file  = f"{CACHE_DIR}/faiss_index_{len(passages)}.bin"
    embeds_file = f"{CACHE_DIR}/embeddings_{len(passages)}.npy"
    model_tag   = model_name.replace("/", "_")

    embed_model = SentenceTransformer(model_name)

    if os.path.exists(index_file) and os.path.exists(embeds_file):
        print("  Loading FAISS index from cache...")
        index = faiss.read_index(index_file)
        return index, embed_model

    print(f"  Encoding {len(passages):,} passages with {model_name} ...")
    embeddings = embed_model.encode(
        passages,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Normalise for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, index_file)
    np.save(embeds_file, embeddings)
    print(f"  FAISS index built: {index.ntotal:,} vectors, dim={dim}")

    return index, embed_model
