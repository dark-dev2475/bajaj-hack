import logging 
from typing import List, Dict, Any, Optional


from clients import openai_async_client,pinecone_client,openai_client 


from search.translator import translate_to_english_async
from search.embedding import _async_embed_query
from search.pinecone_query import _async_pinecone_query
# from search.pinecone_query import _sync_pinecone_query
from search.re_ranker import _rerank_results
from search.embedding import _sync_embed_query
from search.translator import translate_to_english_sync

from query_parser.schema import PolicyQuery


async def perform_search_async(
    raw_query: str,
    index_name: str,
    namespace: str,
    pinecone_client: pinecone_client,
    openai_client: openai_async_client,
    top_k: int = 3,
    structured_query: Optional[PolicyQuery] = None
) -> List[Dict[str, Any]]:
    """
    Asynchronously embeds the query and runs a Pinecone search.
    Optionally reranks results using structured_query.
    Returns a list of matches or a dict with an 'error' key if something fails.
    """
    try:
        query = await translate_to_english_async(raw_query, openai_client)
    except Exception as e:
        logging.error(f"[Async] Translation failed: {e}")
        return [{"error": f"Translation failed: {e}"}]

    vec = await _async_embed_query(query, openai_client)
    if isinstance(vec, dict) and "error" in vec:
        return [vec]

    results = await _async_pinecone_query(vec, index_name, namespace, top_k)

    if structured_query and isinstance(results, list) and results and "error" not in results[0]:
        results = _rerank_results(results, structured_query)

    return results



# this is the sync method currently unused can be used if we use any cli

def perform_search(
    raw_query: str,
    index_name: str,
    namespace: Optional[str] = None,
    top_k: int = 5,
    structured_query: Optional[PolicyQuery] = None
) -> List[Dict[str, Any]]:
    """
    Synchronously embeds the query and runs a Pinecone search.
    Optionally reranks results using structured_query.
    """
    if not openai_client or not pinecone_client:
        logging.error("Sync clients not initialized.")
        return [{"error": "Sync clients not initialized."}]

    try:
        query = translate_to_english_sync(raw_query, openai_client)
    except Exception as e:
        logging.error(f"[Sync] Translation failed: {e}")
        return [{"error": f"Translation failed: {e}"}]

    vec = _sync_embed_query(query)
    if isinstance(vec, dict) and "error" in vec:
        return [vec]

    results = _sync_pinecone_query(vec, index_name, namespace, top_k)
    if structured_query and isinstance(results, list) and results and "error" not in results[0]:
        results = _rerank_results(results, structured_query)

    return results
