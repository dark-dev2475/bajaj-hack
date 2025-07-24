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
    Creates a detailed prompt, now including the language of the user's query.
    """
    context = ""
    for i, result in enumerate(search_results):
        context += f"Source {i+1} (from file: {result['metadata']['source']}):\n"
        context += f"\"{result['metadata']['text']}\"\n\n"

    # --- UPDATED PROMPT ---
    prompt_template = f"""
    You are an expert insurance claim analyst. Your task is to analyze a user's query and a set of relevant policy clauses to make a coverage decision.

    **Policy Clauses (Context in English):**
    ---
    {context}
    ---

    **User's Query (in {query_language.upper()}):** ---
    "{raw_query}"
    ---

    **Your Instructions:**
    Analyze the context to answer the user's query. Your final output MUST be a single, valid JSON object with the specified keys and data types. Do not include any other text.
    - "Decision": Must be one of "Covered", "Not Covered", or "Partial Coverage".
    - "PayoutAmount": Must be an integer. If not applicable, use 0.
    - "Confidence": A float between 0.0 and 1.0.
    - "Justifications": A list of objects. Each object must have lowercase keys "source" (string) and "text" (string), citing the exact text from the context.
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