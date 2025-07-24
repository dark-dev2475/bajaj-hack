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

def _create_prompt(raw_query: str, search_results: List[Dict[str, Any]]) -> str:
    """
    Creates a new, more explicit prompt to prevent validation errors.
    """
    context = ""
    for i, result in enumerate(search_results):
        context += f"Source {i+1} (from file: {result['metadata']['source']}):\n"
        context += f"\"{result['metadata']['text']}\"\n\n"

    # --- NEW, MORE EXPLICIT PROMPT TEMPLATE ---
    prompt_template = f"""
    You are an expert insurance claim analyst. Your task is to analyze a user's query and a set of relevant policy clauses to make a coverage decision.

    **Policy Clauses (Context):**
    ---
    {context}
    ---

    **User's Query:**
    ---
    "{raw_query}"
    ---

    **Your Instructions:**
    Your final output MUST be a single, valid JSON object that strictly adheres to the specified keys and data types. Do not include any other text or explanation.

    **JSON OUTPUT RULES:**
    1.  `Decision`: (string) Must be one of "Covered", "Not Covered", or "Partial Coverage".
    2.  `PayoutAmount`: (integer) The payout amount. **If coverage is denied or the amount is not specified, this value MUST be 0.** Do not use strings.
    3.  `Confidence`: (float) A float between 0.0 and 1.0.
    4.  `Justifications`: (list of objects) A list where each object MUST have three keys, all in lowercase:
        - `source`: (string) The source file name from the context (e.g., "CHOTGDP23004V012223.pdf").
        - `text`: (string) The exact text quote from the context that supports the decision.
        - `relevance`: (float) A float score between 0.0 and 1.0 indicating how relevant this snippet is to the decision.
    
   Example `Justifications` format:
    `"Justifications": [ 
        {{"source": "somefile.pdf", "text": "the relevant quote...", "relevance": 0.87}},
        ...
    ]`
    """
    return prompt_template

def generate_answer(raw_query: str, search_results: List[Dict[str, Any]]) -> Optional[FinalAnswer]:
    if not openai_client:
        return None
    
    prompt = _create_prompt(raw_query, search_results)
    logging.info("Sending final prompt to LLM for decision making...")

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        llm_output_str = response.choices[0].message.content
        
        # Validate the output against our Pydantic schema
        validated_answer = FinalAnswer.model_validate_json(llm_output_str)
        return validated_answer

    except APIError as e:
        logging.error(f"OpenAI API error during answer generation: {e}")
        return None
    except Exception as e:
        logging.error(f"Failed to validate LLM response against schema: {e}")
        return None