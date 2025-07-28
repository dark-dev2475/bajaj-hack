# ml_service/answering_pipeline.py

import logging
import time
import asyncio
from typing import List, Optional

# Import the new, improved, LangChain-powered functions
from search.search_runner import perform_search_async
from answer.answer_generator import generate_answer_async
from answer.answer_schema import FinalAnswer

async def run_single_question_pipeline(
    question: str,
    namespace_id: str,
    index_name: str
) -> Optional[FinalAnswer]:
    """
    Runs the full RAG pipeline for a single question.
    1. Search for relevant documents using the self-query retriever.
    2. Generate a structured answer based on the retrieved context.
    
    Args:
        question: The user's question.
        namespace_id: The Pinecone namespace for the document.
        index_name: The name of the Pinecone index.

    Returns:
        A FinalAnswer object or None if the process fails.
    """
    try:
        # Step 1: Retrieve relevant documents using our advanced search function.
        # This now handles translation, self-querying, and metadata filtering internally.
        logging.info(f"Running search for question: '{question[:50]}...'")
        search_results = await perform_search_async(
            raw_query=question,
            index_name=index_name,
            namespace=namespace_id
        )

        # The search function returns an error dict on failure.
        if not search_results or "error" in search_results[0]:
            logging.warning(f"Search returned no results or an error for question: '{question[:50]}...'")
            # Create a default "not found" answer to maintain a consistent output structure.
            return FinalAnswer(
                Decision="Not Found",
                Reasoning="No relevant information could be found in the document to answer this question.",
                PayoutAmount=None,
                Confidence=0.5,
                Justifications=[]
            )

        # Step 2: Generate a structured answer using the retrieved context.
        # This now uses a reliable LangChain structured output chain.
        logging.info(f"Generating answer for question: '{question[:50]}...'")
        final_answer = await generate_answer_async(
            raw_query=question,
            search_results=search_results,
            query_language="en" # The query is translated to English in the search step
        )

        return final_answer

    except Exception as e:
        logging.exception(f"A critical error occurred in the pipeline for question '{question[:50]}...': {e}")
        return None


async def generate_answers(
    questions: List[str],
    namespace_id: str,
    index_name: str
) -> List[Optional[FinalAnswer]]:
    """
    Performs the full RAG pipeline for a list of questions in parallel.

    Args:
        questions: A list of user questions.
        namespace_id: The Pinecone namespace for the document.
        index_name: The name of the Pinecone index.

    Returns:
        A list of FinalAnswer objects, or None for any question that failed.
    """
    start_time = time.time()
    logging.info(f"Starting parallel answer generation for {len(questions)} questions.")

    # Create a task for each question to run the pipeline concurrently.
    tasks = [
        run_single_question_pipeline(q, namespace_id, index_name)
        for q in questions
    ]
    
    # asyncio.gather runs all tasks and collects their results.
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    logging.info(f"Finished processing all {len(questions)} questions in {elapsed:.2f}s.")
    
    return results
