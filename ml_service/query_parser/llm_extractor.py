# ml_service/query_parser/llm_extractor.py

import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# --- 1. Import the existing schema ---
# This is the correct way to do it, avoiding code duplication.
# We assume the schema is in a file named 'schema.py' in the same directory.
from .schema import PolicyQuery


# --- 2. Initialize the Chat Model ---
# We initialize the LLM that will perform the extraction.
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0 # Use low temperature for predictable, structured output
)

# --- 3. Create the Extraction Chain ---
# This chain binds the LLM to the imported PolicyQuery schema,
# forcing it to output valid JSON that matches your model.
structured_llm = llm.with_structured_output(PolicyQuery)


# --- 4. Create the Prompt ---
# This prompt is now much simpler because LangChain handles the JSON formatting instructions.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at extracting information from user queries into a structured JSON format. Extract all fields from the user's query based on the provided schema."),
    ("human", "{query}")
])


# --- 5. Combine into a Final Chain ---
extraction_chain = prompt | structured_llm


# --- 6. The Main Async Function ---
async def extract_with_llm_async(query: str) -> Optional[dict]:
    """
    Asynchronously extracts structured data from a query using a LangChain extraction chain.

    Args:
        query: The user query string.

    Returns:
        A dictionary of the extracted data, or None if extraction fails.
    """
    if not query or not query.strip():
        return None
        
    logging.info(f"Extracting from query with LangChain: '{query}'")
    try:
        # Invoke the chain with the user's query.
        # LangChain handles the API call, JSON parsing, and validation.
        response_model = await extraction_chain.ainvoke({"query": query})
        
        # Convert the Pydantic model back to a dictionary for the next step in your pipeline.
        return response_model.dict()
        
    except Exception as e:
        logging.exception(f"An error occurred during LangChain extraction: {e}")
        return None

