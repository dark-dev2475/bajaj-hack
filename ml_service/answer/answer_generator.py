# ml_service/answer_generator.py

import os
import logging
from typing import Optional, List, Dict, Any
import asyncio

# --- UPDATED IMPORTS ---
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from .answer_schema import FinalAnswer, Justification
# --- END UPDATED IMPORTS ---


# --- 1. Initialize LLMs and Chains ---
# Using OpenAI's GPT model for better performance
llm = ChatOpenAI(
    model="gpt-4-1106-preview",  # Using GPT-4 Turbo for better accuracy
    temperature=0.1,  # Low temperature for more focused and precise responses
    streaming=True
)

# --- Chain 1: Structured Answer Generation ---
structured_parser = PydanticOutputParser(pydantic_object=FinalAnswer)
structured_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert insurance claims assistant with deep knowledge of policy analysis.

OBJECTIVE:
- Analyze the provided policy clauses thoroughly
- Provide clear, accurate, and well-reasoned decisions based on policy terms
- Consider all relevant clauses when making decisions
- Be precise and specific in your reasoning

GUIDELINES:
- Always cite specific policy clauses that support your decision
- Consider both coverage inclusions and exclusions
- Explain any limitations or conditions that apply
- Be direct and unambiguous in your decision

{format_instructions}
"""),
    ("human", """
POLICY CLAUSES TO ANALYZE:
{context}

USER QUERY ({query_language}):
"{raw_query}"

Please provide a structured analysis based on these policy clauses.
""")
]).partial(format_instructions=structured_parser.get_format_instructions())
answer_chain = structured_prompt | llm | structured_parser


# --- Chain 2: Summarization ---
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a precise and concise insurance response summarizer.

TASK:
Create a 2-3 sentence summary that includes:
1. The final decision (covered/not covered/partially covered)
2. The key reason(s) for the decision
3. Any critical conditions or limitations

Be direct and use simple language that clients can easily understand.
"""),
    ("human", "Please summarize this insurance claim response:\n{full_response_text}")
])
summary_chain = summary_prompt | llm | StrOutputParser()


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
) -> Dict[str, Any]:
    """
    Generates a structured answer and its summary using LangChain chains.
    Returns both the full structured answer and a concise summary.
    """
    logging.info(f"Generating answer and summary for query: '{raw_query}'")
    context = _format_context(search_results)
    
    try:
        # --- Step 1: Generate the full, structured answer ---
        final_answer = await answer_chain.ainvoke({
            "context": context,
            "raw_query": raw_query,
            "query_language": query_language.upper()
        })

        # --- Step 2: Generate a concise summary from the full answer ---
        # --- FIX: Correctly access attributes from the Pydantic model ---
        full_text_for_summary = (
            f"Decision: {final_answer.Decision}. "
            f"Reasoning: {final_answer.Reasoning}"
        )
        # --- END OF FIX ---

        summary = await summary_chain.ainvoke({"full_response_text": full_text_for_summary})

        return {
            "summary": summary,
            "full_response": final_answer.model_dump()
        }

    except Exception as e:
        logging.exception(f"A critical error occurred during answer generation: {e}")
        return {
            "summary": "An error occurred while processing your query.",
            "full_response": None
        }
