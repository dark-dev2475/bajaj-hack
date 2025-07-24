# ml_service/query_parser/main_parser.py
import logging
from typing import Optional
from .llm_extractor import extract_with_llm
from .regex_extractor import extract_with_rules
from .schema import PolicyQuery
from pydantic import ValidationError

def get_structured_query(query: str) -> Optional[PolicyQuery]:
    """
    Hybrid parser: First uses regex to extract fields, then fills missing fields via LLM.
    Regex-extracted fields are always trusted over LLM.
    """
    logging.info(f"Starting hybrid parsing for query: '{query}'")

    # Step 1: Extract high-confidence fields using regex
    regex_data, remaining_query = extract_with_rules(query)
    logging.info(f"Regex extracted: {regex_data}. Remaining query: '{remaining_query}'")

    # Step 2: Identify which fields are missing
    missing_fields = [
        field for field in PolicyQuery.__fields__
        if field not in regex_data or regex_data[field] is None
    ]

    llm_data = {}
    if missing_fields and remaining_query.strip():
        # Step 3: Use LLM only to extract missing fields
        raw_llm_data = extract_with_llm(remaining_query)
        logging.info(f"Raw LLM extracted: {raw_llm_data}")

        # Step 4: Only retain values for fields that regex did NOT already fill
        llm_data = {
            key: value
            for key, value in raw_llm_data.items()
            if key in missing_fields and value is not None
        }
        logging.info(f"Filtered LLM data (excluding regex fields): {llm_data}")

    # Step 5: Final merge — always prioritize regex values
    final_data = {**llm_data, **regex_data}  # regex overwrites llm if same keys

    # Step 6: Validate against schema
    try:
        final_query = PolicyQuery(**final_data)
        logging.info("Successfully parsed and validated full query.")
        return final_query
    except ValidationError as e:
        logging.error(f"Final data invalid against schema: {e}")
        return None
