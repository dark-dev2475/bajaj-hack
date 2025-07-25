# ml_service/answer_generator.py
import os
import json
import logging
from openai import OpenAI, APIError
from typing import Optional, List, Dict, Any

from config import OPENAI_CHAT_MODEL
from answer.answer_schema import FinalAnswer

# Initialize client
try:
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except TypeError:
    logging.error("OPENAI_API_KEY not found.")
    openai_client = None

def _create_prompt(raw_query: str, search_results: List[Dict[str, Any]], query_language: str) -> str:
    """
    Creates a detailed prompt, now including language context and relevance scoring in justifications.
    """
    context = ""
    for i, result in enumerate(search_results):
        source = result['metadata'].get('source', 'unknown')
        text = result['metadata'].get('text', '')
        context += f"Source {i+1} (from file: {source}):\n\"{text.strip()}\"\n\n"

    prompt_template = f"""
You are an expert insurance claim analyst. Your task is to analyze a user's query and a set of relevant policy clauses to make a coverage decision.

---
**Policy Clauses (in English):**
{context}
---

**User's Query (in {query_language.upper()}):**
"{raw_query.strip()}"
---

**Your Instructions:**
Carefully examine the policy clauses and determine whether the user's query is covered. Return a single, valid JSON object with the following keys (and no other output):

- "Decision": One of "Covered", "Not Covered", or "Partial Coverage".
- "PayoutAmount": An integer. Use 0 if no payout is applicable.
- "Confidence": A float between 0.0 and 1.0 representing how confident you are in your decision.
- "Justifications": A list of objects. Each object must contain:
  - "source" (string): Reference to the source file or clause.
  - "text" (string): A short excerpt from the policy clause that supports your decision.
  - "relevance" (float): A number between 0.0 and 1.0 showing how strongly this clause supports your decision. A value closer to 1 means highly relevant.

Respond only with the JSON object. Do not include any preamble, explanations, or extra formatting.
"""
    return prompt_template

def generate_answer(raw_query: str, search_results: List[Dict[str, Any]], query_language: str) -> Optional[FinalAnswer]:
    """
    Generates a structured JSON answer, passing the query language to the prompt.
    """
    if not openai_client:
        return None
    
    # Pass the detected language to the prompt creator
    prompt = _create_prompt(raw_query, search_results, query_language)
    logging.info("Sending final prompt (with language context) to LLM...")

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        llm_output_str = response.choices[0].message.content
        validated_answer = FinalAnswer.model_validate_json(llm_output_str)
        return validated_answer
    except (APIError, Exception) as e:
        logging.error(f"Error during final answer generation: {e}")
        return None