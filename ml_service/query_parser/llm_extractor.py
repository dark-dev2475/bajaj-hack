# ml_service/query_parser/llm_extractor.py
"""
Uses an LLM to extract complex, ambiguous information from a query.
"""
import os
import json
import logging
from openai import OpenAI, APIError
from typing import Dict, Any
from config import OPENAI_CHAT_MODEL

# Initialize the OpenAI client
try:
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except TypeError:
    logging.error("OPENAI_API_KEY not found. Please set the environment variable.")
    openai_client = None

def extract_with_llm(query: str) -> Dict[str, Any]:
    """
    Sends the (potentially simplified) query to an LLM for structured extraction.

    Args:
        query: The user query string.

    Returns:
        A dictionary of extracted data.
    """
    if not openai_client:
        logging.error("OpenAI client not initialized.")
        return {}

    system_prompt = """
    You are an expert assistant. Your task is to extract key information from a user's query 
    about an insurance policy and output it as a valid JSON object. Do not make up information.

    The JSON object must conform to this schema:
    - policy_type: string
    - procedure_or_claim: string
    - location: string (City name)
    - event_details: string
    
    If a value is not mentioned, omit the key from the JSON.
    """
    
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"}
        )
        extracted_json = response.choices[0].message.content
        return json.loads(extracted_json)
    except APIError as e:
        logging.error(f"OpenAI API error during extraction: {e}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from LLM response: {extracted_json}")
        return {}