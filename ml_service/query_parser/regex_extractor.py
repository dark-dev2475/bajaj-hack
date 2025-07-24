# ml_service/query_parser/regex_extractor.py

"""
Extracts high-confidence, simple patterns from a query using regular expressions.
"""
import re
from typing import Dict, Any, Tuple, List

# A list of common Indian cities for simple matching
CITIES = ['delhi', 'mumbai', 'bangalore', 'lucknow', 'kolkata', 'chennai', 'hyderabad', 'pune', 'jaipur', 'bhopal']

# A list of tuples defining rules: (field_name, regex_pattern, type_converter)
REGEX_RULES: List[Tuple[str, str, type]] = [
    # Claimant Age
    ('claimant_age', r'\b(?:i am|i\'m|me is|my age is|age is|aged|currently|about)?\s*(\d{1,2})\s*(?:years old|yrs old|yrs|y/o|yo)?\b', int),
    ('claimant_age', r'\b(\d{1,2})\s*(?:years old|yrs old|yrs|y/o|yo)\b', int),
    ('claimant_age', r'\b(?:age)\s*[:\-]?\s*(\d{1,2})\b', int),

    # Gender
    ('claimant_gender', r'\b(?:i am|i\'m|gender is)?\s*(male|female|man|woman|girl|boy)\b', str),

    # Policy Duration in Months
    ('policy_duration_months', r'\b(?:policy (?:duration|active|is|for))\s*(\d{1,2})\s*(?:months|month)\b', int),
    ('policy_duration_months', r'\b(\d{1,2})\s*(?:months|month)\s*(?:old|active)\b', int),

    # Location
    ('location', r'\b(?:in|at|from|near|around)?\s*(' + '|'.join(CITIES) + r')\b', str),

    # Event Date (Optional)
    ('event_date', r'\b(?:on\s+)?((?:last|next)?\s*(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|year|january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2}(?:st|nd|rd|th)?\s+\w+))\b', str),
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
        if field in extracted_data:
            continue

        match = re.search(pattern, remaining_query, re.IGNORECASE)
        if match:
            raw_value = match.group(1).strip()

            # Normalize gender values
            if field == 'claimant_gender':
                raw_value = raw_value.lower()
                if raw_value in ['man', 'boy']:
                    value = 'male'
                elif raw_value in ['woman', 'girl']:
                    value = 'female'
                else:
                    value = raw_value
            else:
                value = converter(raw_value)

            extracted_data[field] = value
            remaining_query = remaining_query.replace(match.group(0), '', 1).strip()
    
    return extracted_data, remaining_query
