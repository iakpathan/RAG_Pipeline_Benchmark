"""
evaluator.py — Custom scorer using Groq directly (no RAGAS dependency)

Why: RAGAS internally requests n>1 completions which Groq free tier rejects.
Instead we implement the same 3 metrics ourselves using single Groq calls:

  faithfulness     — are all answer claims supported by the retrieved context?
  answer_relevancy — does the answer actually address the question asked?
  context_precision — are the retrieved chunks relevant to the question?
"""

import json
import os
import time
import re
from pipelines import generate_answer
from groq import Groq

RESULTS_DIR = "results"


def _score(client, prompt: str, model: str) -> float:
    """Ask Groq to return a score 0.0-1.0. Retries once on failure."""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            text = resp.choices[0].message.content.strip()
            # Extract first float found in response
            match = re.search(r"\d+\.?\d*", text)
            if match:
                val = float(match.group())
                return min(val, 1.0) if val <= 1.0 else val / 10.0
        except Exception as e:
            print(f"      Retry {attempt+1}: {e}")
            time.sleep(3)
    return 0.5  # fallback if all retries fail


def score_faithfulness(client, question: str, answer: str, contexts: list, model: str) -> float:
    """
    Faithfulness: are the claims in the answer grounded in the context?
    Score 0 = answer contradicts or ignores context, 1 = fully grounded.
    """
    ctx = "\n".join(f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts[:3]))
    prompt = f"""You are evaluating RAG system output.

Context:
{ctx}

Question: {question}
Answer: {answer}

Score the FAITHFULNESS of the answer: how well is it supported by the context above?
- 1.0 = every claim in the answer is directly supported by the context
- 0.5 = answer is partially supported, some claims not in context
- 0.0 = answer contradicts context or ignores it entirely

Reply with ONLY a single decimal number between 0.0 and 1.0."""
    return _score(client, prompt, model)


def score_answer_relevancy(client, question: str, answer: str, model: str) -> float:
    """
    Answer relevancy: does the answer address what was actually asked?
    Score 0 = completely off-topic, 1 = directly and fully answers the question.
    """
    prompt = f"""You are evaluating RAG system output.

Question: {question}
Answer: {answer}

Score the RELEVANCY of the answer to the question:
- 1.0 = answer directly and completely addresses the question
- 0.5 = answer is somewhat related but misses key aspects
- 0.0 = answer is off-topic or doesn't address the question at all

Reply with ONLY a single decimal number between 0.0 and 1.0."""
    return _score(client, prompt, model)


def score_context_precision(client, question: str, contexts: list, model: str) -> float:
    """
    Context precision: are the retrieved chunks actually useful for answering?
    Score 0 = retrieved chunks are noise, 1 = all chunks are highly relevant.
    """
    ctx = "\n".join(f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts[:3]))
    prompt = f"""You are evaluating a retrieval system.

Question: {question}

Retrieved context chunks:
{ctx}

Score the PRECISION of the retrieved context:
- 1.0 = all retrieved chunks are directly relevant to answering the question
- 0.5 = some chunks are relevant, some are noise
- 0.0 = retrieved chunks are irrelevant to the question

Reply with ONLY a single decimal number between 0.0 and 1.0."""
    return _score(client, prompt, model)


def run_evaluation(
    pipelines: dict,
    queries: list,
    ground_truth: list,
    groq_api_key: str,
    llm_model: str = "llama-3.1-8b-instant",
) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    client     = Groq(api_key=groq_api_key)
    all_scores = {}

    for pipeline_name, pipeline_fn in pipelines.items():
        print(f"\n  Running pipeline: {pipeline_name}")

        scores_faith   = []
        scores_relev   = []
        scores_prec    = []
        outputs        = []

        for i, (query, gt) in enumerate(zip(queries, ground_truth)):
            print(f"    Query {i+1}/{len(queries)}: {query[:55]}...")

            # Step 1: retrieve
            result   = pipeline_fn(query)
            contexts = result["contexts"]

            # Step 2: generate answer
            answer = generate_answer(query, contexts, groq_api_key, llm_model)
            time.sleep(1.5)

            # Step 3: score all three metrics (3 separate single-call requests)
            f = score_faithfulness(client, query, answer, contexts, llm_model)
            time.sleep(1.0)
            r = score_answer_relevancy(client, query, answer, llm_model)
            time.sleep(1.0)
            p = score_context_precision(client, query, contexts, llm_model)
            time.sleep(1.0)

            scores_faith.append(f)
            scores_relev.append(r)
            scores_prec.append(p)

            print(f"      faith={f:.2f}  relevancy={r:.2f}  precision={p:.2f}")

            outputs.append({
                "question": query,
                "answer":   answer,
                "contexts": contexts,
                "scores":   {"faithfulness": f, "answer_relevancy": r, "context_precision": p},
            })

        avg = {
            "faithfulness":      round(sum(scores_faith) / len(scores_faith), 3),
            "answer_relevancy":  round(sum(scores_relev) / len(scores_relev), 3),
            "context_precision": round(sum(scores_prec)  / len(scores_prec),  3),
        }
        all_scores[pipeline_name] = avg
        print(f"\n  {pipeline_name} averages: {avg}")

        with open(f"{RESULTS_DIR}/{pipeline_name.replace(' ', '_')}_outputs.json", "w") as f_:
            json.dump(outputs, f_, indent=2)

        print("  Waiting 20s before next pipeline...")
        time.sleep(20)

    with open(f"{RESULTS_DIR}/scores.json", "w") as f_:
        json.dump(all_scores, f_, indent=2)

    return all_scores