# ml_service/query_parser/llm_extractor.py

import logging
from typing import Optional, Dict, Any, Union
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser

# Import our client and OpenAI components
from clients import openai_async_client
from langchain_openai import ChatOpenAI

# Import the schema
from .schema import PolicyQuery

# Configure logging
logger = logging.getLogger(__name__)

# --- LLM Configuration ---
def get_active_llm() -> BaseChatModel:
    """
    Returns an optimized OpenAI LLM instance configured for accurate query extraction.
    Uses GPT-3.5-turbo with settings optimized for structured data extraction.
    """
    if not openai_async_client:
        raise RuntimeError("OpenAI client not configured. Please set OPENAI_API_KEY in your environment.")
    
    logger.info("Initializing OpenAI for extraction with optimized settings")
    return ChatOpenAI(
        model="gpt-4-1106-preview",  # Upgrade to GPT-4 for even better accuracy
        temperature=0,  # Zero temperature for maximum consistency
        timeout=30,  # Increased timeout for reliability
        max_retries=3,  # Added retries for reliability
        client=openai_async_client,
        model_kwargs={
            "response_format": {"type": "json_object"}  # Enforce JSON output
        }
    )


# --- Create the extraction prompt ---
EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise insurance query analyzer specialized in extracting structured data.

TASK:
Extract structured information from user queries following these guidelines:

KEY FIELDS TO IDENTIFY:
1. Policy Type:
   - Identify specific insurance category (health, life, vehicle, etc.)
   - Note any policy-specific codes or numbers

2. Claimant Details:
   - Age (convert text to numbers)
   - Gender (standardize to M/F/Other)
   - Any pre-existing conditions

3. Claim Details:
   - Claim amount if mentioned
   - Date of claim/incident
   - Type of claim (reimbursement, cashless, etc.)

4. Location Information:
   - City/State/Country
   - Hospital/Institution name if relevant

5. Policy Duration:
   - Start date
   - End date
   - Duration in years

6. Event Details:
   - Incident description
   - Related medical procedures
   - Involved parties

IMPORTANT:
- Extract ONLY facts present in the query
- Do NOT make assumptions or add information
- Leave fields empty if information is not provided
- Ensure all output is in valid JSON format
- Be precise with numbers and dates

For each extracted field, provide a confidence score:
- 1.0: Explicitly stated in the query
- 0.8: Strongly implied with context
- 0.5: Inferred with some uncertainty
- 0.0: Not found or highly uncertain

Include these scores in your JSON output.

{format_instructions}
"""),
    ("human", """Please analyze this insurance query and extract all relevant information:
{query}""")
])

def create_extraction_chain():
    """
    Creates a new extraction chain using the active LLM.
    This allows switching LLMs without restarting the application.
    """
    try:
        # Get the currently active LLM
        llm = get_active_llm()
        
        # Create a structured output chain
        structured_llm = llm.with_structured_output(PolicyQuery)
        
        # Combine prompt with LLM
        chain = EXTRACTION_PROMPT | structured_llm
        
        logger.info(f"Created extraction chain using {llm.__class__.__name__}")
        return chain
    except Exception as e:
        logger.error(f"Failed to create extraction chain: {e}")
        raise


# Create the initial extraction chain
extraction_chain = create_extraction_chain()

async def extract_with_llm_async(query: str) -> Optional[Dict[str, Any]]:
    """
    Asynchronously extracts structured data from a query using OpenAI.

    Args:
        query: The user query string.

    Returns:
        A dictionary of the extracted data, or None if extraction fails.
        The dictionary follows the PolicyQuery schema.
    """
    if not query or not query.strip():
        logger.warning("Empty query received")
        return None
        
    # Limit query size to prevent excessive token usage
    MAX_QUERY_LENGTH = 4000  # Adjust based on your needs
    if len(query) > MAX_QUERY_LENGTH:
        query = query[:MAX_QUERY_LENGTH]
        logger.warning(f"Query truncated to {MAX_QUERY_LENGTH} characters")
    
    logger.info(f"Extracting structured data from query: '{query[:100]}...'")
    
    try:
        # Use the extraction chain with error handling
        response_model = await extraction_chain.ainvoke({"query": query})
        result = response_model.model_dump()
        
        # Add validation
        if result.get('policy_type'):
            valid_types = {'health', 'life', 'motor', 'property', 'travel'}
            if result['policy_type'].lower() not in valid_types:
                logger.warning(f"Unusual policy type detected: {result['policy_type']}")
        
        if result.get('claim_amount'):
            try:
                float(str(result['claim_amount']).replace(',', ''))
            except ValueError:
                logger.warning(f"Invalid claim amount format: {result['claim_amount']}")
        
        return result
        
    except Exception as e:
        logger.exception(f"Failed to extract data from query: {e}")
        return None

def switch_llm() -> None:
    """
    Switches to a different LLM and recreates the extraction chain.
    Call this if you want to change LLMs at runtime.
    """
    global extraction_chain
    extraction_chain = create_extraction_chain()

