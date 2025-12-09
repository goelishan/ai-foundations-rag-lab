import json
import time
import os
import sys
from typing import List,Dict

print("--- rag_eval.py script started ---") # Added for debugging

# Add the project root to sys.path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.answer_builder import answer_question

TEST_QUESTIONS_FILE = "/content/ai-foundations-rag-lab/tests/questions/test_questions.txt"
RESULTS_FILE = "/content/ai-foundations-rag-lab/tests/results/test_results.json"

def load_test_questions(path:str)-> List[str]:
  print("--- Loading test questions ---") # Added for debugging
  with open(path,"r",encoding="utf-8") as f:
    questions=[line.strip() for line in f.readlines() if line.strip()]
  return questions

def evaluate_rag_system(questions:str,top_k:int=5):
  print("--- Starting RAG system evaluation ---") # Added for debugging
  results=[]

  for q in questions:
    start_time=time.time()

    print(f"\nRunning RAG System on Question - {q}\n")
    result=answer_question(q,top_k=top_k)
    answer=result["answer"]
    sources=result["sources"]

    end_time=time.time()
  
    failed=False
    reason=""

    if "I dont know" in answer:
      failed=True
      reason="LLM lacked context for answer."
    elif len(sources)==0:
      failed=True
      reason="Retriever returned no sources."
    elif not any(f"[Source" in answer for _ in sources):
      failed=True
      reason="No citation found in answer."
    
    results.append({
      "question":q,
      "answer":answer,
      "sources":sources,
      "time_taken_in_sec":round(end_time-start_time,2),
      "failed":failed,
      "failure_reason":reason
    })

  return results

def save_results(results:List[Dict],path:str=RESULTS_FILE):
  print("--- Saving results ---") # Added for debugging
  with open(path,"w",encoding="utf-8")as f:
    json.dump(results,f,ensure_ascii=True,indent=2)
  
  print(f"Results saved to path - {path}")


def result_summary(results:List[Dict]):
  print("--- Generating result summary ---") # Added for debugging
  total=len(results)
  failure_list=[r for r in results if r["failed"]]
  success=total-len(failure_list)

  print("\n-----------RAG Evaluation Summary----------\n")
  print(f"Total Questions - {total}")
  print(f"Successful Answers - {success}")
  print(f"Failed Answers - {len(failure_list)}")
  print(f"\nFailures Breakdown-\n")
  for f in failure_list:
    print(f"Q: {f["question"]}\nA: {f["failure_reason"]}") # Fixed: use 'f' instead of 'failure'
  print("--------------------------------------------------------")

if __name__=="__main__":
    print("--- Entering main execution block ---") # Added for debugging
    questions=load_test_questions(TEST_QUESTIONS_FILE)
    print(f"Loaded {len(questions)} questions.")
    results=evaluate_rag_system(questions=questions,top_k=5)
    save_results(results)
    result_summary(results)
