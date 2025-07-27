# ml_service/query_parser/llm_extractor.py

import os
import json
import logging
from openai import AsyncOpenAI, APIError
from typing import Dict, Any
from config import OPENAI_CHAT_MODEL

# Initialize the async OpenAI client
try:
    openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except TypeError:
    logging.error("OPENAI_API_KEY not found. Please set the environment variable.")
    openai_client = None

async def extract_with_llm_async(query: str) -> Dict[str, Any]:
    """
    Asynchronously sends the query to an LLM for structured extraction.

    Args:
        query: The user query string.

    Returns:
        A dictionary of extracted data.
    """
    if not openai_client:
        logging.error("OpenAI client not initialized.")
        return {}

    system_prompt = """
You are a highly reliable insurance assistant.

Given a user's natural language query, extract and return structured information in strict JSON format based on the following schema:

{
  "policy_type": string,
  "claimant_age": integer | null,
  "claimant_gender": string | null,
  "procedure_or_claim": string,
  "location": string,
  "policy_duration_months": integer | null,
  "event_details": string
}

### Extraction Rules:
- Do **NOT** assume or infer any value not clearly stated in the query.
- If a value is not mentioned, set it to `null`, except for `event_details`, which should always summarize the key event.
- Only include exact city names in `location` (no addresses).
- Use lowercase for strings unless proper noun (e.g. city).
- For `claimant_gender`, use "male", "female", or `null`.
- `procedure_or_claim` should be one key medical term or reason for claim (e.g., "fracture", "hospitalization", "surgery").
- For `policy_duration_months`, extract number of months if specified like "3 months old", "valid for 1 year" → 12, etc.

### Example Output:
{
  "policy_type": "personal accident",
  "claimant_age": 32,
  "claimant_gender": "male",
  "procedure_or_claim": "fracture",
  "location": "Lucknow",
  "policy_duration_months": 3,
  "event_details": "fractured arm at home"
}
"""


    try:
        response = await openai_client.chat.completions.create(
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
