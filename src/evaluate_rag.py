import os
import sys
import pandas as pd
import asyncio

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import ask
from langchain_ollama import ChatOllama

# --- EVALUATION DATASET ---
test_set = [
    {
        "question": "What is the notice period for terminating the supply contract?",
        "ground_truth": "The contractual relationship can be terminated by either party with one month's notice in writing."
    },
    {
        "question": "What happens if a customer does not allow the meter reading?",
        "ground_truth": "DEW21 can estimate the consumption based on the last reading or the consumption of comparable customers if unable to determine actual consumption."
    },
    {
        "question": "What is SCHUFA used for in energy contracts?",
        "ground_truth": "SCHUFA is used to check the creditworthiness (Bonitätsprüfung) of customers."
    }
]

async def judge_with_reasoning(question, context, answer, ground_truth, criterion):
    """
    A more granular LLM Judge that provides a score AND a reason.
    """
    judge = ChatOllama(model="llama3", temperature=0)
    
    if criterion == "faithfulness":
        prompt = f"""Evaluate if the provided 'Answer' is supported by the 'Context'. 
        If the answer contains info NOT in the context, score 0. If it perfectly matches context, score 1.
        
        CONTEXT: {context}
        ANSWER: {answer}
        
        Return your response strictly in this format: 
        SCORE: [0 or 1]
        REASON: [Short explanation why]"""
    else: # relevance
        prompt = f"""Compare the 'Answer' to the 'Ground Truth' and 'Question'.
        Does the answer provide the correct information requested? 
        If it's mostly correct, score 1. If it's wrong or irrelevant, score 0.
        
        QUESTION: {question}
        GROUND TRUTH: {ground_truth}
        ANSWER: {answer}
        
        Return your response strictly in this format:
        SCORE: [0 or 1]
        REASON: [Short explanation why]"""

    try:
        res = judge.invoke(prompt).content
        
        # Parse score
        score = 0.0
        if "SCORE: 1" in res.upper(): score = 1.0
        elif "SCORE: 0" in res.upper(): score = 0.0
        
        # Parse reason
        reason = "Unknown"
        if "REASON:" in res.upper():
            reason = res.upper().split("REASON:")[1].strip()
            
        return score, reason
    except Exception as e:
        return 0.5, f"Error: {e}"

async def run_evaluation():
    print("🚀 Starting DEW21 Quality Audit with AI Reasoning...")
    
    results = []
    for item in test_set:
        print(f"🔍 Auditing: {item['question']}")
        rag_output = ask(item["question"], chat_history=[], lang="en")
        
        f_score, f_reason = await judge_with_reasoning(
            item["question"], rag_output["contexts"], rag_output["answer"], item["ground_truth"], "faithfulness"
        )
        r_score, r_reason = await judge_with_reasoning(
            item["question"], rag_output["contexts"], rag_output["answer"], item["ground_truth"], "relevance"
        )
        
        results.append({
            "Question": item["question"],
            "Faithfulness": f_score,
            "F_Reason": f_reason,
            "Relevance": r_score,
            "R_Reason": r_reason,
            "Answer": rag_output["answer"][:100] + "..."
        })

    # 🏆 Detailed Scorecard
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("🏆 DEW21 AI QUALITY SCORECARD (Annotated)")
    print("="*80)
    for index, row in df.iterrows():
        print(f"\n❓ Q: {row['Question']}")
        print(f"📊 Faith: {row['Faithfulness']} | Reason: {row['F_Reason']}")
        print(f"📊 Relev: {row['Relevance']} | Reason: {row['R_Reason']}")
    
    print("\n" + "="*80)
    print(f"📣 Avg. Faithfulness: {df['Faithfulness'].mean():.2%}")
    print(f"📣 Avg. Relevance:    {df['Relevance'].mean():.2%}")
    print("="*80)
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/eval_annotated.csv", index=False)
    print(f"✅ Full annotated report saved to: data/eval_annotated.csv")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
