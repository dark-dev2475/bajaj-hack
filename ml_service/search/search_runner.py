# search/search_runner.py

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
import asyncio

# --- UPDATED IMPORTS ---
from clients import openai_async_client, pinecone_client
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from .translator import translate_to_english_async

def normalize_vector(vector: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Normalize a vector to unit length using L2 normalization.
    
    Args:
        vector: Input vector as list or numpy array
        
    Returns:
        Normalized vector as numpy array
    """
    if isinstance(vector, list):
        vector = np.array(vector)
    
    # Handle zero vectors
    norm = np.linalg.norm(vector)
    if norm == 0:
        logger.warning("Zero vector encountered during normalization")
        return vector
    
    return vector / norm

def normalize_vectors_batch(vectors: List[List[float]]) -> List[np.ndarray]:
    """
    Normalize a batch of vectors to unit length.
    
    Args:
        vectors: List of input vectors
        
    Returns:
        List of normalized vectors
    """
    return [normalize_vector(vec) for vec in vectors]

# --- 1. Initialize Core Components using OpenAI ---
# Initialize embeddings with the latest model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    client=openai_async_client
)

# Initialize LLM for re-ranking with optimized settings
rerank_llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",  # Latest model with better JSON handling
    temperature=0.0,  # Maximum precision for re-ranking
    client=openai_async_client,
    model_kwargs={
        "response_format": {"type": "json_object"}  # Enforce JSON output
    }
)
# --- END OF UPDATE ---


# --- 2. Define the Re-ranking Logic ---
reranker_prompt_template = """
You are an expert insurance policy analysis system specializing in semantic relevance scoring.

TASK:
Score the relevance of each document to the user's query on a scale of 0 to 1, where:
- 1.0: Directly answers the query with exact policy details
- 0.8-0.9: Highly relevant, contains most of the needed information
- 0.5-0.7: Partially relevant, contains some useful information
- 0.1-0.4: Tangentially related
- 0.0: Not relevant at all

SCORING CRITERIA:
1. Policy Coverage Match: How well the clause matches the type of coverage in question
2. Specific Details: Presence of relevant numbers, dates, conditions
3. Context Alignment: How well the context matches the user's scenario
4. Information Completeness: Amount of relevant information provided

FORMAT:
Return ONLY a JSON object with this exact structure:
{{"scores": [number, number, ...]}}

USER QUERY: "{query}"

DOCUMENTS TO SCORE:
{documents}
"""
reranker_prompt = PromptTemplate.from_template(reranker_prompt_template)

# This chain will take the query and documents and output a JSON object with scores.
reranker_chain = reranker_prompt | rerank_llm | JsonOutputParser()


async def perform_search_async(
    raw_query: str,
    index_name: str,
    namespace: str,
    top_k: int = 3, # The final number of chunks to return
) -> List[Dict[str, Any]]:
    """
    Performs a two-step search:
    1. A broad vector search to retrieve a set of candidate documents.
    2. An LLM-based re-ranking step to find the most relevant documents from the candidates.
    """
    logging.info(f"Performing 2-step search for: '{raw_query}' in namespace '{namespace}'")
    
    try:
        # Use OpenAI for translation
        english_query = await translate_to_english_async(raw_query, rerank_llm)
        logging.info(f"Translated query to English: '{english_query}'")

        # --- Direct Pinecone Query using OpenAI embeddings ---
        # 1. Get and normalize the query embedding vector
        raw_query_vector = await embeddings.aembed_query(english_query)
        query_vector = normalize_vector(raw_query_vector).tolist()  # Convert back to list for Pinecone
        
        # Log embedding stats for debugging
        logger.debug(f"Query vector norm before normalization: {np.linalg.norm(raw_query_vector):.4f}")
        logger.debug(f"Query vector norm after normalization: {np.linalg.norm(query_vector):.4f}")

        # 2. Get the Pinecone index object
        index = pinecone_client.Index(index_name)

        # 3. Perform the search with normalized vectors
        search_response = await asyncio.to_thread(
            index.query,
            vector=query_vector,
            top_k=10,  # Retrieve more documents for re-ranking
            namespace=namespace,
            include_metadata=True
        )

        # 4. Manually format the results
        candidate_docs = [
            {
                "metadata": match.metadata,
                "chunk_text": match.metadata.get("text", "") # Ensure text is present
            }
            for match in search_response.matches
        ]
        
        if not candidate_docs:
            logging.warning("Initial vector search returned no results.")
            return []
        
        logging.info(f"Retrieved {len(candidate_docs)} candidate documents for re-ranking.")

        # --- Step 2: LLM Re-ranking (Precision) ---
        docs_for_reranking = "\n\n".join(
            [f"Document {i+1}:\n{doc['chunk_text']}" for i, doc in enumerate(candidate_docs)]
        )
        
        response = await reranker_chain.ainvoke({
            "query": english_query,
            "documents": docs_for_reranking
        })
        scores = response.get("scores", [])

        if len(scores) != len(candidate_docs):
            logging.error("Re-ranker score count does not match document count. Returning top candidates.")
            return candidate_docs[:top_k]

        scored_docs = sorted(
            zip(scores, candidate_docs),
            key=lambda x: x[0],
            reverse=True
        )

        final_results = [doc for score, doc in scored_docs[:top_k]]
        logging.info(f"Re-ranked and selected top {len(final_results)} documents.")
        return final_results

    except Exception as e:
        logging.exception(f"An error occurred during the search and re-rank process: {e}")
        return [{"error": "Failed to perform search."}]
