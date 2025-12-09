# answer_builder.py
"""
Answer Builder for RAG system (Google Colab compatible)
- Correct OpenAI SDK usage
- No leading indentation in prompt
- Sort retrieved passages by score
- Clean, strict instructions for grounded answering
"""

import os
import sys
from typing import List, Dict

# Ensure root path is in sys.path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.retriever import retrieve
from openai import OpenAI
import httpx # Explicitly import httpx for client initialization

LLM_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------
# Prompt Constructor
# ---------------------------------------------------------
def build_prompt(question: str, passages: List[Dict]) -> str:
    """
    Build clean RAG prompt without indentation issues.
    """

    context_lines = []
    for i, p in enumerate(passages, start=1):
        context_lines.append(
            f"[Source {i}] (doc={p['doc_id']} | passage={p['passage_id']})\n{p['text']}\n"
        )

    context_block = "\n".join(context_lines)

    # NO LEADING TABS OR SPACES → Critical for LLM correctness
    prompt = (
        "You are an expert assistant. Use ONLY the provided context passages to answer.\n\n"
        f"QUESTION:\n{question}\n\n"
        "CONTEXT:\n"
        f"{context_block}\n"
        "INSTRUCTIONS:\n"
        "- If the answer is not present in the passages, say EXACTLY: \"I don't know from the provided documents.\"\n"
        "- Cite passages using [Source X].\n"
        "- Do NOT add external knowledge.\n"
    )

    return prompt


# ---------------------------------------------------------
# Main RAG Answer Pipeline
# ---------------------------------------------------------
def answer_question(question: str, top_k: int = 6, model_name: str = LLM_MODEL):
    """
    Retrieves relevant passages → Builds prompt → Queries LLM → Returns grounded answer.
    """

    passages = retrieve(question, top_k=top_k)

    if not passages:
        return {"answer": "No relevant documents found.", "sources": []}

    # Sort by similarity score
    passages = sorted(passages, key=lambda x: x["score"], reverse=True)

    prompt = build_prompt(question, passages)

    # Create OpenAI client (correct method to avoid proxy issues)
    client = OpenAI(http_client=httpx.Client()) # Pass an explicit httpx.Client()

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500,
    )

    # Correct way to access the message content attribute
    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": passages,
    }


# ---------------------------------------------------------
# Test Run
# ---------------------------------------------------------
if __name__ == "__main__":
    q = "How have hybrid engines changed Formula 1 strategy?"
    result = answer_question(q)

    print("\n---- FINAL ANSWER ----\n")
    print(result["answer"])

    print("\n---- SOURCES USED ----\n")
    for s in result["sources"]:
        print(f"- {s['doc_id']} | {s['passage_id']} | score={s['score']:.4f}")
