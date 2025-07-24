# ml_service/search.py
"""
Handles the semantic search and retrieval logic for the application.
"""
import os
import logging
import pinecone
from openai import OpenAI
from query_parser.schema import PolicyQuery
from typing import List, Dict, Any
from langdetect import detect

# Initialize clients from environment variables
try:
    pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except TypeError:
    logging.error("API keys not found. Please set environment variables.")
    pc = None
    openai_client = None


def _translate_query_if_needed(query: str) -> str:
    """Detects language and translates to English if necessary."""
    try:
        if detect(query) != 'en':
            logging.info(f"Translating non-English query to English.")
            response = openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": f"Translate the following text to English: \"{query}\""}]
            )
            return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error during translation, using original query: {e}")
    return query


def _rerank_results(results: List[Dict[str, Any]], structured_query: PolicyQuery) -> List[Dict[str, Any]]:
    """
    Refines search results by boosting scores of chunks containing keywords.
    """
    logging.info("Re-ranking search results for precision...")
    
    # Extract keywords from the structured query to check for
    keywords = [structured_query.procedure_or_claim, structured_query.location]
    keywords = [k.lower() for k in keywords if k] # Filter out None values and convert to lowercase

    if not keywords:
        logging.info("No keywords for re-ranking. Returning original results.")
        return results

    for match in results:
        text_lower = match['metadata']['text'].lower()
        bonus = 0.0
        for keyword in keywords:
            if keyword in text_lower:
                bonus += 0.1  # Add a small score bonus for each keyword match
        match['score'] += bonus
    
    # Sort the results again based on the new, boosted score
    return sorted(results, key=lambda x: x['score'], reverse=True)

def perform_search(raw_query: str, structured_query: PolicyQuery, index_name: str) -> List[Dict[str, Any]]:
    if not pc or not openai_client:
        logging.error("Clients not initialized.")
        return []

    # --- NEW: Translate the query first ---
    query_for_embedding = _translate_query_if_needed(raw_query)
    
    # 1. Embed the (now guaranteed English) query
    logging.info("Embedding the user query...")
    query_embedding = openai_client.embeddings.create(
        input=[query_for_embedding],
        model="text-embedding-3-small"
    ).data[0].embedding

    # (The rest of the function for querying and re-ranking remains the same)
    logging.info(f"Querying vector DB...")
    index = pc.Index(index_name)
    results = index.query(
        vector=query_embedding,
        top_k=3, 
        include_metadata=True
    )
    
    return results['matches'] # For simplicity, returning direct results for now.