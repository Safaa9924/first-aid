"""
================================================================================
 STAGE 07 · PROMPTING & GENERATION
 First Aid Reference Guide (St. John Ambulance Canada) — RAG Pipeline
================================================================================
Builds the grounded, structured first-aid prompt and sends it to an LLM
through OpenRouter (https://openrouter.ai). The API key is NEVER hardcoded
here — it is read from the environment / Streamlit secrets at call time,
so it is safe to commit and share this file.

Required setup (pick one):
    - Environment variable : OPENROUTER_API_KEY
    - Streamlit secrets    : st.secrets["OPENROUTER_API_KEY"]

Usage:
    from importlib import import_module
    prompting = import_module("07_prompting")
================================================================================
"""

import os
import time
import requests
from dotenv import load_dotenv
# ==================================================================
# OpenRouter configuration
# ==================================================================


# Load environment variables
load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

DEFAULT_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "qwen/qwen3-8b"
)

TEMPERATURE = 0.1
MAX_TOKENS = 1200
SEED = 42
GROUNDING_THRESHOLD = 1.5


def get_openrouter_api_key():
    """
    Resolve the OpenRouter API key without ever hardcoding it in source.
    Priority: Streamlit secrets -> environment variable.
    """

    try:
        import streamlit as st
        if "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass

    return os.environ.get("OPENROUTER_API_KEY")


# ==================================================================
# Stage 17 — Prompt Builder
# ==================================================================

def build_chat_prompt(condition: str, context: str) -> str:

    if not context or not context.strip():
        raise ValueError("Retrieved context is empty.")

    prompt = f"""You are an expert Evidence-Based First Aid Assistant.
Answer the user question strictly in ENGLISH using ONLY the provided context. Never add outside knowledge.
If the answer is not in the context, output: "I couldn't find this information in the retrieved first aid reference."

CORE RULES:
1. Concise & Direct: Max 5 bullet points per section. No introductions or summaries.
2. No Repetition: Mention each piece of advice only once.
3. Omit Empty Sections: If a section has no context data, skip its header completely.
4. Clean Markdown: Strictly follow the structure below.

STRUCTURE TO FOLLOW:
## First Aid: {condition}

## Immediate Actions
- [Essential steps, max 5 items]

## Avoid
- [Warnings directly related, max 5 items]

## When to Call Emergency Services
- [Specific situations]

## Additional Notes
- [Extra crucial info, if any]

## Evidence Source
- [Source/Organization name from context, or 'Retrieved first aid reference document']

============================
USER QUESTION: {condition}
============================
RETRIEVED CONTEXT:
{context}
"""
    return prompt


# ==================================================================
# Stage 18 — LLM Generation (OpenRouter)
# ==================================================================

def generate_answer(
    prompt,
    model=DEFAULT_MODEL,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    seed=SEED,
):
    api_key = get_openrouter_api_key()

    if not api_key:
        return (
            "No OpenRouter API key found. Set the OPENROUTER_API_KEY "
            "environment variable or add it to Streamlit secrets."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }

    start = time.time()

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()

        elapsed = time.time() - start
        result = response.json()

        answer = result["choices"][0]["message"]["content"].strip()

        if not answer:
            answer = "The language model returned an empty response."

        print("=" * 60)
        print("LLM GENERATION STATS")
        print("=" * 60)
        print("Model            :", result.get("model", model))
        print(f"Inference Time   : {elapsed:.2f} sec")
        usage = result.get("usage", {})
        print("Completion Tokens:", usage.get("completion_tokens", "N/A"))

        return answer

    except Exception as e:
        print("=" * 60)
        print("OPENROUTER ERROR")
        print("=" * 60)
        print(e)
        return f"The first-aid assistant could not reach OpenRouter: {e}"


# ==================================================================
# Stage 19 — Confidence / grounding report
# ==================================================================

def confidence_label(score, best):
    if best <= 0:
        return "Low"

    ratio = score / best

    if ratio >= 0.80:
        return "High"
    elif ratio >= 0.50:
        return "Medium"
    return "Low"


def build_grounding_report(context):
    """context: dict returned by 06_retrieve_context.build_context_package"""

    selected_df = context["selected_df"]

    if selected_df.empty:
        return {
            "num_sources": 0,
            "best_score": 0.0,
            "sources": [],
        }

    best_score = selected_df["rerank_score"].max()

    sources = []
    for rank, (_, row) in enumerate(
        selected_df.sort_values("rerank_score", ascending=False).iterrows(), start=1
    ):
        sources.append({
            "rank": rank,
            "chunk_id": row["chunk_id"],
            "rerank_score": float(row["rerank_score"]),
            "confidence": confidence_label(row["rerank_score"], best_score),
            "text": row["chunk_text"],
        })

    return {
        "num_sources": len(sources),
        "best_score": float(best_score),
        "sources": sources,
    }


# ==================================================================
# End-to-end convenience wrapper
# ==================================================================

def answer_first_aid_question(user_question, context, translate_back=None):
    """
    Build the prompt from the retrieved context, call OpenRouter, and
    (optionally) translate the answer back to the user's language.

    translate_back: callable(text) -> str, e.g. 06_retrieve_context.translate_to_arabic
    """

    prompt = build_chat_prompt(condition=user_question, context=context["context_text"])
    english_answer = generate_answer(prompt)

    final_answer = (
        translate_back(english_answer) if translate_back and context.get("language") == "ar"
        else english_answer
    )

    grounding = build_grounding_report(context)

    return {
        "english_answer": english_answer,
        "final_answer": final_answer,
        "grounding": grounding,
    }
