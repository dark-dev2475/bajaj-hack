# ml_service/answer_generator.py

import os
import logging
from typing import Optional, List, Dict, Any
import asyncio

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from .answer_schema import FinalAnswer, Justification # Assuming schema is in this file

# --- 1. Initialize the Chat Model for Answer Generation ---
# We create a specific LLM chain for this task.
# Using a higher temperature (e.g., 0.2) can allow for more natural-sounding text.
llm = ChatOpenAI(
    model="gpt-4-turbo", # Using a more powerful model for the final answer is often a good idea
    temperature=0.2,
     # ✅ Explicitly set this
)

# --- 2. Create the Answer Generation Chain ---
# This chain will take the context and question, and force the LLM
# to output a response that perfectly matches your FinalAnswer schema.
structured_llm = llm.with_structured_output(FinalAnswer, method="function_calling")

# --- 3. Define the System and Human Prompts ---
# This template is cleaner and more maintainable than a large f-string.
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an empathetic and highly accurate insurance claims assistant.
Your task is to analyze the provided policy clauses and the user's query to produce a structured JSON response.

- Base your decision exclusively on the provided policy clauses.
- Provide clear, step-by-step reasoning for your decision.
- If you cannot determine a specific value (like PayoutAmount) from the text, leave it as null.
"""),
    ("human", """
---
Policy Clauses:
{context}
---
User Query ({query_language}): "{raw_query}"
---
""")
])

# --- 4. Combine into a Final Chain ---
answer_chain = prompt | structured_llm

# --- Helper function to format the context ---
def _format_context(search_results: List[Dict[str, Any]]) -> str:
    """Formats the search results into a single string for the prompt."""
    if not search_results:
        return "No relevant policy clauses were found."
        
    context_blocks = [
        f'Source File: {res.get("metadata", {}).get("source", "unknown")}\nContent: "{res.get("chunk_text", "").strip()}"'
        for res in search_results
    ]
    return "\n\n---\n\n".join(context_blocks)


# --- 5. The Main Answer Generation Function ---
async def generate_answer_async(
    raw_query: str,
    search_results: List[Dict[str, Any]],
    query_language: str,
) -> Optional[FinalAnswer]:
    """
    Generates a structured answer using a LangChain chain.
    """
    logging.info(f"Generating final answer for query: '{raw_query}'")

    # Format the retrieved documents into a single context string.
    context = _format_context(search_results)

    try:
        # Invoke the chain. LangChain handles the API call, JSON parsing,
        # validation, and has built-in retry logic.
        final_answer = await answer_chain.ainvoke({
            "context": context,
            "raw_query": raw_query,
            "query_language": query_language.upper()
        })
        return final_answer
        
    except Exception as e:
        logging.exception(f"A critical error occurred during answer generation: {e}")
        return None

