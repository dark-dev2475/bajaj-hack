# ml_service/answer_generator.py

import os
import logging
from typing import Optional, List, Dict, Any

from openai import APIError
from config import OPENAI_CHAT_MODEL
from answer.answer_schema import FinalAnswer
from clients import openai_async_client
import asyncio


def _create_prompt(raw_query: str, search_results: List[Dict[str, Any]], query_language: str) -> str:
    """
    Builds a structured prompt for the LLM with source-aware policy clause context.
    """
    context_blocks = [
        f'Source {i + 1} (from file: {res["metadata"].get("source", "unknown")}):\n"{res["metadata"].get("text", "").strip()}"'
        for i, res in enumerate(search_results)
    ]
    context = "\n\n".join(context_blocks)

    return f"""You are an empathetic insurance claims assistant. Analyze the user's query and provided policy clauses to produce a structured JSON response.
---
Policy Clauses:
{context}
---
User Query ({query_language.upper()}): "{raw_query.strip()}"
---
Respond with a JSON object matching this schema:
{{
  "Decision": string ("Covered", "Not Covered", "Partial Coverage"),
  "Reasoning": string,
  
  "PayoutAmount": integer,
  "Confidence": float (0.0 - 1.0),
  "Justifications": [
    {{"source": string, "text": string, "relevance": float}}
  ]
}}
Do not include any other text.""".strip()


async def generate_answer_async(
    raw_query: str,
    search_results: List[Dict[str, Any]],
    query_language: str,
    openai_client: Optional[Any] = None,
    max_retries: int = 3,
    timeout: int = 30
) -> Optional[FinalAnswer]:
    """
    Calls the LLM asynchronously to generate a JSON-formatted answer.
    Retries on transient errors and logs raw LLM output on JSON decode errors.
    """
    client = openai_client or openai_async_client
    if client is None:
        logging.error("OpenAI client is not initialized.")
        return None

    prompt = _create_prompt(raw_query, search_results, query_language)
    logging.info("Prompt constructed. Sending to LLM...")

    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=OPENAI_CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=timeout
            )
            content = response.choices[0].message.content
            logging.info("LLM response received. Validating JSON...")
            return FinalAnswer.model_validate_json(content)
        except APIError as api_err:
            logging.error(f"OpenAI API error (attempt {attempt}): {api_err}")
            last_exception = api_err
        except asyncio.TimeoutError:
            logging.error(f"OpenAI LLM call timed out after {timeout} seconds (attempt {attempt})")
            last_exception = "Timeout"
        except Exception as e:
            # Log the raw LLM output if available
            if 'content' in locals():
                logging.error(f"Failed to decode/validate LLM output (attempt {attempt}): {e}\nRaw output: {content}")
            else:
                logging.exception(f"Unexpected error during LLM generation (attempt {attempt})")
            last_exception = e
        await asyncio.sleep(2 * attempt)  # Exponential backoff

    logging.error(f"All attempts to generate answer failed. Last error: {last_exception}")
    return None
