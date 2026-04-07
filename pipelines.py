"""
pipelines.py — Standard RAG, HyDE RAG, and Multi-Query RAG implementations
"""

import re
import numpy as np
import faiss
from groq import Groq


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _embed_query(query: str, embed_model) -> np.ndarray:
    """Encode a single string and L2-normalise it."""
    emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    return emb


def _search(emb: np.ndarray, index, passages: list, k: int) -> list[str]:
    """Return top-k passages for a given query embedding."""
    _, ids = index.search(emb, k)
    return [passages[i] for i in ids[0] if i < len(passages)]


def _call_groq(prompt: str, api_key: str, model: str, max_tokens: int = 512) -> str:
    """Single Groq chat completion call."""
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline A: Standard RAG
# ─────────────────────────────────────────────────────────────────────────────

def standard_rag(
    query: str,
    index,
    embed_model,
    passages: list,
    k: int = 5,
) -> dict:
    """
    Baseline: embed query → retrieve top-k → generate answer.
    Returns dict with 'question', 'answer', 'contexts'.
    """
    q_emb    = _embed_query(query, embed_model)
    contexts = _search(q_emb, index, passages, k)
    return {"question": query, "contexts": contexts}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline B: HyDE RAG
# ─────────────────────────────────────────────────────────────────────────────

def hyde_rag(
    query: str,
    index,
    embed_model,
    passages: list,
    k: int = 5,
    api_key: str = "",
    model: str = "llama-3.1-8b-instant",
) -> dict:
    """
    HyDE (Gao et al. 2022):
      1. Ask LLM to write a hypothetical document that would answer the query.
      2. Embed the hypothetical document (not the raw query).
      3. Retrieve real documents closest to that hypothetical embedding.

    WHY: The hypothetical doc and real answer docs share vocabulary and style,
         so they sit closer in embedding space than a short query would.
    """
    hyde_prompt = (
        f"Write a detailed, factual paragraph (100-150 words) that directly "
        f"answers the following question. Do not include the question itself.\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    hypothetical_doc = _call_groq(hyde_prompt, api_key, model, max_tokens=200)

    h_emb    = _embed_query(hypothetical_doc, embed_model)
    contexts = _search(h_emb, index, passages, k)

    return {
        "question":        query,
        "contexts":        contexts,
        "hypothetical_doc": hypothetical_doc,   # save for inspection
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline C: Multi-Query RAG
# ─────────────────────────────────────────────────────────────────────────────

def multi_query_rag(
    query: str,
    index,
    embed_model,
    passages: list,
    k: int = 5,
    api_key: str = "",
    model: str = "llama-3.1-8b-instant",
) -> dict:
    """
    Multi-Query RAG:
      1. Generate N semantically distinct reformulations of the query.
      2. Retrieve top-k docs for each reformulation independently.
      3. Deduplicate via a seen-set; return merged context.

    WHY: Covers vocabulary mismatches — if the user's phrasing misses the
         document's vocabulary, one of the reformulations may hit it.
    """
    rephrase_prompt = (
        f"Generate exactly 5 different ways to ask the following question. "
        f"Each reformulation should use different vocabulary and perspective. "
        f"Output one reformulation per line, no numbering, no explanation.\n\n"
        f"Original question: {query}"
    )
    raw = _call_groq(rephrase_prompt, api_key, model, max_tokens=300)

    # Parse — one per line, strip empty lines
    variants = [line.strip() for line in raw.split("\n") if line.strip()][:5]
    variants.insert(0, query)   # always include the original

    seen     = set()
    contexts = []
    for variant in variants:
        v_emb = _embed_query(variant, embed_model)
        docs  = _search(v_emb, index, passages, k)
        for doc in docs:
            if doc not in seen:
                seen.add(doc)
                contexts.append(doc)

    # Cap at 2k to avoid overwhelming the LLM context
    contexts = contexts[: k * 2]

    return {
        "question":  query,
        "contexts":  contexts,
        "variants":  variants,   # save for inspection
    }


# ─────────────────────────────────────────────────────────────────────────────
# Answer generator (used by evaluator, same for all pipelines)
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(question: str, contexts: list, api_key: str, model: str) -> str:
    """Given retrieved contexts, generate a final answer with the LLM."""
    context_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts[:5]))
    prompt = (
        f"Answer the question using ONLY the provided context. "
        f"If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    return _call_groq(prompt, api_key, model, max_tokens=300)
