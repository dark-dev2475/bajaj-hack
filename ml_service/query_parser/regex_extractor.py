# ml_service/query_parser/regex_extractor.py
"""
Extracts high-confidence, simple patterns from a query using regular expressions.
"""
import re
from typing import Dict, Any, Tuple, List

# A list of tuples, where each defines a rule: (field_name, pattern, type_converter)
REGEX_RULES: List[Tuple[str, str, type]] = [
    ('claimant_age', r'(\d{1,2})\s*years old', int),
    ('claimant_age', r'age is\s*(\d{1,2})', int),
    ('claimant_age', r'i\'m\s*(\d{1,2})', int),
    # Add more simple, high-confidence rules here
]

def extract_with_rules(query: str) -> Tuple[Dict[str, Any], str]:
    """
    Scans the query for predefined patterns and extracts them.

    Args:
        query: The raw user query string.

    Returns:
        A tuple containing:
        - A dictionary of extracted data.
        - The remaining query string after removing matched patterns.
    """
    extracted_data: Dict[str, Any] = {}
    remaining_query = query

    for field, pattern, converter in REGEX_RULES:
        # Check if the field has already been found by a previous rule
        if field in extracted_data:
            continue

        match = re.search(pattern, remaining_query, re.IGNORECASE)
        if match:
            # Extract the value and convert it to the correct type
            value = converter(match.group(1))
            extracted_data[field] = value
            # Remove the matched text from the query to not confuse the LLM
            remaining_query = remaining_query.replace(match.group(0), '', 1).strip()
    
    return extracted_data, remaining_query