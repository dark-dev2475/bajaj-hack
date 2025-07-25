# ml_service/answer_generator.py
import os
import logging
from typing import Optional, List, Dict, Any

from openai import APIError
from config import OPENAI_CHAT_MODEL
from answer.answer_schema import FinalAnswer
from clients import openai_async_client  # use initialized client


def _create_prompt(raw_query: str, search_results: List[Dict[str, Any]], query_language: str) -> str:
    """
    Builds a structured prompt with context clauses for the LLM.
    """
    context = ""
    for i, result in enumerate(search_results, 1):
        source = result['metadata'].get('source', 'unknown')
        text = result['metadata'].get('text', '').strip()
        context += f"Source {i} (from file: {source}):\n\"{text}\"\n\n"

    return f"""
You are an empathetic insurance claims assistant. Analyze the user's query and provided policy clauses to produce a structured JSON response.
---
Policy Clauses:
{context}
---
User Query ({query_language.upper()}): "{raw_query.strip()}"
---
Respond with a JSON object matching this schema:
{{
  "Decision": string ("Covered","Not Covered","Partial Coverage"),
  "Reasoning": string,
  "NextSteps": string,
  "PayoutAmount": integer,
  "Confidence": float (0.0-1.0),
  "Justifications": [
    {{"source": string, "text": string, "relevance": float}}
  ]
}}
Do not include any other text.""".strip()


async def generate_answer_async(
    raw_query: str,
    search_results: List[Dict[str, Any]],
    query_language: str,
    openai_client: Any = None
) -> Optional[FinalAnswer]:
    """
    Asynchronously calls the LLM to generate a JSON-formatted answer.
    """
    # Use the passed client or fallback to the globally imported one
    client = openai_client or openai_async_client
    if not client:
        logging.error("OpenAI client not initialized.")
        return None

    prompt = _create_prompt(raw_query, search_results, query_language)
    logging.info("Sending prompt to LLM...")

    try:
        response = await client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        llm_output = response.choices[0].message.content
        return FinalAnswer.model_validate_json(llm_output)

    except (APIError, Exception) as e:
        logging.error(f"LLM generation failed: {e}")
        return None
