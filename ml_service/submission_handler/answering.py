import logging
import time
import asyncio
from typing import List
from search import search_runner as search
from answer import answer_generator

from query_parser.main_parser import get_structured_query
from clients import pinecone_client, openai_async_client

async def generate_answers(
    questions: List[str],
    namespace_id: str,
    index_name: str
) -> List[str]:
    """
    Performs semantic search and LLM answer generation in parallel.
    Each task is self-contained and handles its own errors.
    """

    async def _answer(q: str) -> str:
        start = time.time()
        try:
            structured_query = await get_structured_query(q)

            hits = await search.perform_search_async(
                raw_query=q,
                index_name=index_name,
                namespace=namespace_id,
                pinecone_client=pinecone_client,
                openai_client=openai_async_client,
                structured_query=structured_query
            )
            if not hits:
                return "No relevant information found."

            ans = await answer_generator.generate_answer_async(
                raw_query=q,
                search_results=hits,
                query_language="en",
                openai_client=openai_async_client
            )
            result = ans.Reasoning if ans else "Could not determine an answer."

            elapsed = time.time() - start
            logging.info(f"[Timing] Generated answer for '{q[:30]}...' in {elapsed:.2f}s")
            return result

        except Exception as e:
            logging.error(f"[Answer Error] '{q}': {e}")
            return "Error processing this question."

    # Create and run tasks concurrently. Since each task handles its own
    # exceptions and is guaranteed to return a string, we don't need
    # to re-check for exceptions after gathering.
    tasks = [asyncio.create_task(_answer(q)) for q in questions]
    results: List[str] = await asyncio.gather(*tasks)

    return results