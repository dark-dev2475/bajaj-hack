# ml_service/query_parser/main_parser.py
"""
Orchestrates the hybrid query parsing workflow.
"""
import logging
from typing import Optional
from .llm_extractor import extract_with_llm
from .regex_extractor import extract_with_rules
from .schema import PolicyQuery

def get_structured_query(query: str) -> Optional[PolicyQuery]:
    """
    Orchestrates the optimized hybrid parsing of a raw user query.

    Args:
        query: The raw user query string.

    Returns:
        A Pydantic PolicyQuery object, or None if parsing fails.
    """
    logging.info(f"Starting hybrid parsing for query: '{query}'")
    
    # Step 1: Extract high-confidence data with regex
    regex_data, remaining_query = extract_with_rules(query)
    logging.info(f"Regex extracted: {regex_data}. Remaining query: '{remaining_query}'")
    
    # Step 2: Use LLM on the remainder of the query
    llm_data = {}
    if remaining_query:
        llm_data = extract_with_llm(remaining_query)
        logging.info(f"LLM extracted: {llm_data}")
    
    # Step 3: Merge results, giving LLM data precedence for shared keys
    final_data = {**regex_data, **llm_data}
    
    # Step 4: Validate with Pydantic
    try:
        structured_query = PolicyQuery(**final_data)
        logging.info(f"Successfully parsed and validated query.")
        return structured_query
    except Exception as e:
        logging.error(f"Failed to validate final data against schema: {e}")
        return None