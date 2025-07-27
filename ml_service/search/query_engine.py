import logging
from clients import  pinecone_client
from clients import openai_async_client as openai_async_client
import time
from functools import wraps

def timed(name="Function"):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            result = await func(*args, **kwargs)
            end = time.time()
            logging.info(f"[Timing] {name} took {end - start:.2f}s")
            return result
        return wrapper
    return decorator

@timed("Query Engine")
async def query_engine(query: str, namespace_id: str, index_name: str) -> str:
    logging.info(f"[QueryEngine] Querying vector DB with query: {query}")
    relevant_chunks = pinecone_client.search_vector_db(query, namespace_id, index_name)

    if not relevant_chunks:
        logging.warning("[QueryEngine] No relevant chunks found.")
        return "No relevant information found in the documents."

    context = "\n".join(relevant_chunks)
    logging.info("[QueryEngine] Constructed context for LLM call.")

    response = await openai_async_client.get_llm_response(query, context)
    logging.info("[QueryEngine] LLM response received.")

    return response
