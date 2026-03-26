"""
DEW21 RAG Evaluation Suite
===========================
Comprehensive evaluation across 6 key metrics using an LLM-as-Judge approach.

Metrics:
  1. Faithfulness       — Is the answer grounded in the retrieved context?
  2. Answer Relevance   — Does the answer address the question?
  3. Context Precision   — Are the top-ranked retrieved chunks truly relevant?
  4. Context Recall      — Did we retrieve all the information needed?
  5. Correctness         — Does the answer match the expected ground truth?
  6. Hallucination Rate  — Does the answer introduce claims not in the context?

Usage:
  python -m src.evaluate_rag              # Run full evaluation
  python -m src.evaluate_rag --quick      # Run 3-question quick test
"""
import os
import sys
import json
import asyncio
import time
from datetime import datetime
from typing import Any

import pandas as pd  # type: ignore

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import ask_stream, ahybrid_retrieve  # type: ignore
from langchain_ollama import ChatOllama  # type: ignore

# ─────────────────────────────────────────────
# EVALUATION DATASET — 10 diverse questions
# Covers: Electricity GTC, Gas GTC, SCHUFA, Creditreform, Costs
# ─────────────────────────────────────────────
EVAL_DATASET = [
    # ── Electricity GTC ──
    {
        "id": "E1",
        "question": "What is the notice period for terminating the electricity supply contract?",
        "ground_truth": "The contractual relationship can be terminated by either party with one month's notice in writing.",
        "category": "Electricity"
    },
    {
        "id": "E2",
        "question": "Under what conditions can DEW21 interrupt electricity supply due to payment default?",
        "ground_truth": "DEW21 can interrupt supply if the customer's payment default equals twice the monthly installment or at least €100.00 including collection costs, after providing four weeks advance notice.",
        "category": "Electricity"
    },
    {
        "id": "E3",
        "question": "How does DEW21 estimate electricity consumption when a meter reading is not possible?",
        "ground_truth": "DEW21 estimates consumption based on the last reading or the consumption of comparable customers if unable to determine actual consumption.",
        "category": "Electricity"
    },
    # ── Gas GTC ──
    {
        "id": "G1",
        "question": "How often is natural gas consumption billed?",
        "ground_truth": "Natural gas consumption is generally recorded and billed every 12 months, unless an interim or final bill is issued earlier.",
        "category": "Gas"
    },
    {
        "id": "G2",
        "question": "When can DEW21 request an advance payment for gas supply?",
        "ground_truth": "DEW21 can request an advance payment if the customer is in arrears with a payment to a non-insignificant extent, repeatedly defaults within 12 months, has outstanding claims from a previous contract, or in other justified cases.",
        "category": "Gas"
    },
    # ── SCHUFA ──
    {
        "id": "S1",
        "question": "What is SCHUFA used for in energy contracts?",
        "ground_truth": "SCHUFA is used to check the creditworthiness (Bonitätsprüfung) of customers before entering into energy supply contracts.",
        "category": "SCHUFA"
    },
    {
        "id": "S2",
        "question": "How long does SCHUFA store personal data?",
        "ground_truth": "The standard storage period is three years from the date of completion. Inquiry data is deleted after twelve months. Trouble-free contract data is deleted immediately after notification of termination.",
        "category": "SCHUFA"
    },
    # ── Creditreform ──
    {
        "id": "C1",
        "question": "What is the role of Creditreform in the DEW21 context?",
        "ground_truth": "Creditreform processes information on creditworthiness of individuals and companies, storing data about name, address, financial circumstances, liabilities, and payment behavior for creditor protection.",
        "category": "Creditreform"
    },
    # ── Costs ──
    {
        "id": "CO1",
        "question": "How much does electricity reconnection cost during working hours?",
        "ground_truth": "Electricity reconnection during working hours costs €54.58 without VAT or €64.95 with VAT.",
        "category": "Costs"
    },
    # ── Cross-document ──
    {
        "id": "X1",
        "question": "What rights does a customer have regarding their personal data stored by SCHUFA and Creditreform?",
        "ground_truth": "Customers have the right to information (Art. 15 GDPR), rectification (Art. 16), erasure (Art. 17), and restriction of processing (Art. 18). They can also object to processing under Art. 21(1) GDPR for reasons arising from their particular situation.",
        "category": "Cross-doc"
    },
]

QUICK_DATASET = [EVAL_DATASET[i] for i in range(min(3, len(EVAL_DATASET)))]  # For fast testing


# ─────────────────────────────────────────────
# RAG CALL WRAPPER — captures answer + contexts
# ─────────────────────────────────────────────
def run_rag(question: str, lang="en", mode="Standard", k=10) -> dict:
    """Run the RAG pipeline and capture answer + retrieved document texts."""
    full_answer = ""
    retrieved_docs = []
    
    for chunk in ask_stream(question, chat_history=[], lang=lang, 
                            retrieved_docs_out=retrieved_docs, mode=mode, k=k):
        full_answer += chunk
    
    contexts = []
    for d in retrieved_docs:
        contexts.append({
            "text": d.page_content,
            "source": d.metadata.get("source", d.metadata.get("doc_name", "Unknown")),
        })
    
    return {
        "answer": full_answer,
        "contexts": contexts,
        "context_text": "\n\n".join([c["text"] for c in contexts]),
        "sources": list(set(c["source"] for c in contexts)),
    }


# ─────────────────────────────────────────────
# LLM JUDGE — Evaluates a single criterion
# ─────────────────────────────────────────────
JUDGE_PROMPTS = {
    "faithfulness": """You are a strict evaluation judge. Evaluate whether the ANSWER is fully supported by the CONTEXT.
Every claim in the answer must be traceable to the context. If the answer contains ANY information not present in the context, score it lower.

CONTEXT:
{context}

ANSWER:
{answer}

Score on a scale of 0.0 to 1.0 (use increments of 0.1):
- 1.0 = Every claim is directly supported by context
- 0.7 = Mostly supported, minor unsupported details
- 0.5 = Partially supported
- 0.3 = Mostly unsupported
- 0.0 = Completely fabricated

Respond ONLY in this exact JSON format:
{{"score": <float>, "reason": "<one sentence explanation>"}}""",

    "relevance": """You are a strict evaluation judge. Evaluate whether the ANSWER directly and completely addresses the QUESTION.

QUESTION:
{question}

ANSWER:
{answer}

Score on a scale of 0.0 to 1.0:
- 1.0 = Fully addresses the question with precise, complete information
- 0.7 = Addresses the question but misses some details
- 0.5 = Partially relevant
- 0.3 = Tangentially related
- 0.0 = Completely irrelevant

Respond ONLY in this exact JSON format:
{{"score": <float>, "reason": "<one sentence explanation>"}}""",

    "correctness": """You are a strict evaluation judge. Compare the ANSWER to the GROUND TRUTH.
The answer does not need to be word-for-word identical, but must convey the same key facts.

QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

ANSWER:
{answer}

Score on a scale of 0.0 to 1.0:
- 1.0 = Contains all key facts from ground truth
- 0.7 = Contains most key facts
- 0.5 = Contains some key facts
- 0.3 = Contains few key facts
- 0.0 = Wrong or contradicts ground truth

Respond ONLY in this exact JSON format:
{{"score": <float>, "reason": "<one sentence explanation>"}}""",

    "context_precision": """You are a strict evaluation judge. Evaluate whether the top retrieved CONTEXT chunks are relevant to answering the QUESTION.
Focus on the first 5 chunks — are they actually useful for answering this specific question?

QUESTION:
{question}

TOP CONTEXT CHUNKS:
{context}

Score on a scale of 0.0 to 1.0:
- 1.0 = All top chunks are highly relevant
- 0.7 = Most chunks are relevant, some noise
- 0.5 = Mixed — about half relevant
- 0.3 = Mostly irrelevant chunks
- 0.0 = No relevant chunks retrieved

Respond ONLY in this exact JSON format:
{{"score": <float>, "reason": "<one sentence explanation>"}}""",

    "context_recall": """You are a strict evaluation judge. Given the GROUND TRUTH answer, evaluate whether the retrieved CONTEXT contains enough information to reconstruct that answer.

GROUND TRUTH:
{ground_truth}

CONTEXT:
{context}

Score on a scale of 0.0 to 1.0:
- 1.0 = Context contains ALL information needed to derive the ground truth
- 0.7 = Context contains most of the needed information
- 0.5 = Context contains some of the needed information
- 0.3 = Context barely contains relevant information
- 0.0 = Context is completely missing the needed information

Respond ONLY in this exact JSON format:
{{"score": <float>, "reason": "<one sentence explanation>"}}""",

    "hallucination": """You are a strict evaluation judge. Identify whether the ANSWER contains any claims, facts, or details that are NOT present in the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Score the HALLUCINATION RATE on a scale of 0.0 to 1.0:
- 0.0 = No hallucination — every claim is in the context (BEST)
- 0.3 = Minor hallucination — small unsupported details
- 0.5 = Moderate hallucination
- 0.7 = Significant hallucination
- 1.0 = Entirely hallucinated (WORST)

Respond ONLY in this exact JSON format:
{{"score": <float>, "reason": "<one sentence explanation>"}}""",
}


async def judge_criterion(question, context, answer, ground_truth, criterion) -> dict:
    """Use LLM to judge a single criterion. Returns {score, reason}."""
    judge = ChatOllama(model="qwen2.5:7b", temperature=0)
    
    prompt_template = JUDGE_PROMPTS[criterion]
    prompt = prompt_template.format(
        question=question,
        context=context[:4000],  # Truncate to avoid context overflow
        answer=answer,
        ground_truth=ground_truth,
    )
    
    try:
        resp = await judge.ainvoke(prompt)
        text = resp.content.strip()
        
        # Parse JSON from response
        # Handle cases where LLM wraps in ```json ... ```
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        result = json.loads(text.strip())
        return {
            "score": float(result.get("score", 0.0)),
            "reason": result.get("reason", "No reason provided"),
        }
    except Exception as e:
        return {"score": 0.0, "reason": f"Judge error: {e}"}


# ─────────────────────────────────────────────
# MAIN EVALUATION RUNNER
# ─────────────────────────────────────────────
async def evaluate(dataset: list[dict[str, Any]] | None = None, verbose: bool = True, k=10):
    """Run the full evaluation suite for a specific k."""
    if dataset is None:
        dataset = EVAL_DATASET
    
    print("\n" + "=" * 70)
    print(f"🚀 DEW21 RAG EVALUATION SUITE (k={k})")
    print(f"   {len(dataset)} questions • 6 metrics • LLM-as-Judge")
    print("=" * 70)
    
    all_results = []
    total_start = time.time()
    
    for item in dataset:
        qid = item["id"]
        question = item["question"]
        ground_truth = item["ground_truth"]
        category = item.get("category", "General")
        
        print(f"\n{'─' * 50}")
        print(f"🔍 [{qid}] {question}")
        print(f"   Category: {category}")
        
        # 1. Run RAG
        t0 = time.time()
        rag_result = run_rag(question, k=k)
        latency = time.time() - t0
        print(f"   ⏱  RAG latency: {latency:.1f}s | Sources: {len(rag_result['sources'])}")
        
        # 2. Judge all 6 criteria in parallel
        criteria = ["faithfulness", "relevance", "correctness", 
                     "context_precision", "context_recall", "hallucination"]
        
        judge_tasks = [
            judge_criterion(
                question, rag_result["context_text"], 
                rag_result["answer"], ground_truth, c
            ) for c in criteria
        ]
        
        judgments = await asyncio.gather(*judge_tasks)
        
        # 3. Collect results
        row: dict[str, Any] = {
            "ID": qid,
            "Category": category,
            "Question": question,
            "Latency_s": round(float(latency), 1),  # type: ignore
            "Answer_Preview": rag_result["answer"][:120] + "...",
            "Num_Sources": len(rag_result["sources"]),
            "K_Value": k,
        }
        
        for criterion, judgment in zip(criteria, judgments):
            row[f"{criterion}_score"] = judgment["score"]
            row[f"{criterion}_reason"] = judgment["reason"]
        
        all_results.append(row)
        
        if verbose:
            print(f"   📊 Faith: {row['faithfulness_score']:.1f} | "
                  f"Relev: {row['relevance_score']:.1f} | "
                  f"Correct: {row['correctness_score']:.1f} | "
                  f"Ctx-P: {row['context_precision_score']:.1f} | "
                  f"Ctx-R: {row['context_recall_score']:.1f} | "
                  f"Halluc: {row['hallucination_score']:.1f}")
    
    total_time = time.time() - total_start
    
    # ─── SCORECARD ───
    df = pd.DataFrame(all_results)
    
    score_cols = [c for c in df.columns if c.endswith("_score")]
    
    print("\n" + "=" * 70)
    print("🏆 DEW21 RAG EVALUATION SCORECARD")
    print("=" * 70)
    
    print("\n📊 Overall Averages:")
    print("─" * 40)
    for col in score_cols:
        metric_name = col.replace("_score", "").replace("_", " ").title()
        avg = df[col].mean()
        bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
        emoji = "✅" if avg >= 0.7 else "⚠️" if avg >= 0.5 else "❌"
        # For hallucination, lower is better
        if "hallucination" in col:
            emoji = "✅" if avg <= 0.3 else "⚠️" if avg <= 0.5 else "❌"
        print(f"  {emoji} {metric_name:22s} {bar} {avg:.1%}")
    
    print(f"\n⏱  Total evaluation time: {total_time:.0f}s")
    print(f"   Average latency per question: {df['Latency_s'].mean():.1f}s")
    
    # ─── Per-Category Breakdown ───
    print("\n📋 Per-Category Breakdown:")
    print("─" * 40)
    for cat in df["Category"].unique():
        cat_df = df[df["Category"] == cat]
        avg_correct = cat_df["correctness_score"].mean()
        avg_faith = cat_df["faithfulness_score"].mean()
        print(f"  {cat:15s} | Correctness: {avg_correct:.0%} | Faithfulness: {avg_faith:.0%} | n={len(cat_df)}")
    
    # ─── Save Results ───
    os.makedirs("evaluation", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Full detailed CSV
    csv_path = f"evaluation/eval_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Full report saved to: {csv_path}")
    
    # Summary JSON
    summary: dict[str, Any] = {
        "timestamp": timestamp,
        "num_questions": len(dataset) if dataset else 0,
        "total_time_s": round(float(total_time), 1),  # type: ignore
        "avg_latency_s": round(float(df["Latency_s"].mean()), 1),  # type: ignore
        "scores": {},
    }
    for col in score_cols:
        metric = col.replace("_score", "")
        summary["scores"][metric] = {
            "mean": round(float(df[col].mean()), 3),  # type: ignore
            "min": round(float(df[col].min()), 3),  # type: ignore
            "max": round(float(df[col].max()), 3),  # type: ignore
        }
    
    json_path = f"evaluation/eval_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Summary saved to: {json_path}")
    
    # Also save as latest
    df.to_csv("evaluation/eval_latest.csv", index=False)
    with open("evaluation/eval_latest.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ Evaluation complete!")
    print("=" * 70)
    
    return df, summary


# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DEW21 RAG Evaluation Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick 3-question test")
    parser.add_argument("--sweep", action="store_true", help="Run a sweep across different k values mapping from k=1 to k=20")
    args = parser.parse_args()
    
    dataset = QUICK_DATASET if args.quick else EVAL_DATASET
    
    if args.sweep:
        k_values = [1, 5, 10, 20]
        print(f"🎯 Starting K-Sweep Analysis: {k_values}")
        all_sweep_dfs = []
        for k_val in k_values:
            # Explicitly unpack to avoid Coroutine indexing errors in static analysis
            df_k, _ = asyncio.run(evaluate(dataset, k=k_val)) # type: ignore
            all_sweep_dfs.append(df_k)
        
        # Save consolidated sweep report
        sweep_df = pd.concat(all_sweep_dfs)
        sweep_path = "evaluation/eval_sweep_latest.csv"
        sweep_df.to_csv(sweep_path, index=False)
        print(f"\n📊 Consolidated K-Sweep report saved to: {sweep_path}")
        print("✅ K-Sweep complete!")
    else:
        asyncio.run(evaluate(dataset))
