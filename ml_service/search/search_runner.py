# search/search_runner.py

import logging
from typing import List, Dict, Any, Optional
import asyncio

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from clients import pinecone_client, openai_async_client
from .translator import translate_to_english_async

# --- 1. Initialize Core Components ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# A powerful LLM is needed for the re-ranking task
rerank_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)


# --- 2. Define the Re-ranking Logic ---
# --- THIS IS THE FIX ---
# The prompt is updated to be more strict, preventing the LLM from adding comments.
reranker_prompt_template = """
You are an expert at evaluating the relevance of insurance policy clauses.
Based on the user's query, score each of the following document chunks on a scale from 0 to 1 for how relevant it is to answering the query.
Respond **only** with a valid JSON object containing a list of scores, one for each document. Do not include any other text, comments, or explanations.

Example Response: {{"scores": [0.2, 0.9, 0.5]}}

User Query: "{query}"

Documents:
{documents}
"""
# --- END OF FIX ---
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
        english_query = await translate_to_english_async(raw_query, openai_async_client)
        logging.info(f"Translated query to English: '{english_query}'")

        # Get the Pinecone index object from our already initialized client.
        index = pinecone_client.Index(index_name)
        
        # Pass the index object directly to the PineconeVectorStore.
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text", # Specify the metadata field containing the text
            namespace=namespace
        )

        # --- Step 1: Broad Vector Search (Recall) ---
        # Retrieve more documents than needed (e.g., 10) to cast a wide net.
        candidate_docs_lc = await vector_store.asimilarity_search(english_query, k=10)
        
        # Format the results into the dictionary structure our re-ranker expects
        candidate_docs = [
            {
                "metadata": doc.metadata,
                "chunk_text": doc.page_content
            }
            for doc in candidate_docs_lc
        ]

        if not candidate_docs:
            logging.warning("Initial vector search returned no results.")
            return []
        
        logging.info(f"Retrieved {len(candidate_docs)} candidate documents for re-ranking.")

        # --- Step 2: LLM Re-ranking (Precision) ---
        # Format documents for the re-ranking prompt
        docs_for_reranking = "\n\n".join(
            [f"Document {i+1}:\n{doc['chunk_text']}" for i, doc in enumerate(candidate_docs)]
        )
        
        # Get relevance scores from the LLM
        response = await reranker_chain.ainvoke({
            "query": english_query,
            "documents": docs_for_reranking
        })
        scores = response.get("scores", [])

        if len(scores) != len(candidate_docs):
            logging.error("Re-ranker score count does not match document count. Returning top candidates.")
            return candidate_docs[:top_k]

        # Combine documents with their new scores
        scored_docs = sorted(
            zip(scores, candidate_docs),
            key=lambda x: x[0],
            reverse=True
        )

        # Return the top_k documents with the highest relevance scores
        final_results = [doc for score, doc in scored_docs[:top_k]]
        logging.info(f"Re-ranked and selected top {len(final_results)} documents.")
        return final_results

    except Exception as e:
        logging.exception(f"An error occurred during the search and re-rank process: {e}")
        return [{"error": "Failed to perform search."}]
