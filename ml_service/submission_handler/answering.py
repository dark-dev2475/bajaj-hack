# ml_service/submission_handler/answering.py

import logging
import time
import asyncio
from typing import List, Union

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
    Performs semantic search and LLM answer generation in parallel,
    protecting each task so failures don’t cancel the group.
    """

    async def _answer(q: str) -> str:
        start = time.time()
        try:
            structured_query = get_structured_query(q)

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

    # Create tasks
    tasks = [asyncio.create_task(_answer(q)) for q in questions]

    # Gather with return_exceptions=True so no task cancels others
    raw_results: List[Union[str, Exception]] = await asyncio.gather(*tasks, return_exceptions=True)

    # Normalize results: turn any Exception into an error message
    results: List[str] = []
    for idx, res in enumerate(raw_results):
        if isinstance(res, Exception):
            logging.error(f"[Answering] Task for question {idx} raised: {res}")
            results.append("Error processing this question.")
        else:
            results.append(res)

    return results
