# ml_service/query_parser/main_parser.py
import logging
from typing import Optional
import asyncio

# We'll assume the functions you call have async versions
from .llm_extractor import extract_with_llm_async
from .regex_extractor import extract_with_rules
from .schema import PolicyQuery
from pydantic import ValidationError

async def get_structured_query_async(query: str) -> Optional[PolicyQuery]:
    """
    Asynchronously parses a query using a hybrid regex-first, LLM-fallback approach.
    This version is non-blocking and suitable for an async application.
    """
    logging.info(f"Starting async hybrid parsing for query: '{query}'")

    # Step 1: Extract high-confidence fields using regex (CPU-bound, runs in a thread)
    # This is fast, so running it in a thread is optional but good practice.
    regex_data, remaining_query = await asyncio.to_thread(extract_with_rules, query)
    logging.info(f"Regex extracted: {regex_data}. Remaining query for LLM: '{remaining_query}'")

    # Step 2: Identify which fields are still missing
    missing_fields = [
        field for field in PolicyQuery.model_fields # Use model_fields for Pydantic v2
        if field not in regex_data or regex_data.get(field) is None
    ]

    llm_data = {}
    if missing_fields and remaining_query.strip():
        # Step 3: Use LLM to extract only the missing fields (I/O-bound)
        logging.info(f"Calling LLM to extract missing fields: {missing_fields}")
        raw_llm_data = await extract_with_llm_async(remaining_query) # Assumes an async version exists
        
        if raw_llm_data:
            logging.info(f"Raw LLM extracted: {raw_llm_data}")
            # Step 4: Only retain values for fields that regex did NOT already fill
            llm_data = {
                key: value
                for key, value in raw_llm_data.items()
                if key in missing_fields and value is not None
            }
            logging.info(f"Filtered LLM data: {llm_data}")
        else:
            logging.warning("LLM extraction returned no data.")

    # Step 5: Final merge — always prioritize regex values
    # The order here is important: regex_data will overwrite any keys from llm_data.
    final_data = {**llm_data, **regex_data}
    logging.info(f"Final merged data for validation: {final_data}")

    # Step 6: Validate against the Pydantic schema
    try:
        final_query = PolicyQuery(**final_data)
        logging.info(f"Successfully parsed and validated query: {final_query.model_dump_json(indent=2)}")
        return final_query
    except ValidationError as e:
        logging.error(f"Final data failed validation against PolicyQuery schema: {e}")
        return None

