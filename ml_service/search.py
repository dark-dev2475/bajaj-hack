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

# Initialize clients from environment variables
try:
    pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except TypeError:
    logging.error("API keys not found. Please set environment variables.")
    pc = None
    openai_client = None

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
    """
    Embeds a query, runs a search, and refines the results.
    """
    if not pc or not openai_client:
        logging.error("Pinecone or OpenAI client not initialized.")
        return []

    # 1. Embed the query using the same model as docs
    logging.info("Embedding the user query...")
    query_embedding = openai_client.embeddings.create(
        input=[raw_query],
        model="text-embedding-3-small"
    ).data[0].embedding

    # 2. Search logic: Query for nearest neighbors
    # We'll fetch more results (k=10) to give the re-ranker a good pool of candidates.
    logging.info("Querying vector DB with k=10 for initial results...")
    index = pc.Index(index_name)
    initial_results = index.query(
        vector=query_embedding,
        top_k=10, 
        include_metadata=True
    )

    # 3. Refinement
    refined_results = _rerank_results(initial_results['matches'], structured_query)

    # 4. Citations are returned via the metadata field
    # Return the top 5 results after refinement
    return refined_results[:5]