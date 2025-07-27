import logging
from typing import  Any, Optional
import asyncio
from clients import pinecone_client



async def _async_pinecone_query(
    vec: Any,
    index_name: str,
    namespace: str,
    top_k: int
) -> Any:
    """
    Asynchronously query Pinecone with the given vector.
    This runs the synchronous Pinecone query in a thread pool.
    """
    def blocking_query():
     try:
        logging.info(f"Querying Pinecone with: top_k={top_k}, namespace={namespace}")
        index = pinecone_client.Index(index_name)
        response = index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        logging.info(f"Pinecone async response: {response}")
        return response.get("matches", [])
     except Exception as e:
        logging.error(f"[Async] Pinecone query failed: {e}")
        return [{"error": f"Pinecone query failed: {e}"}]


    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, blocking_query)





def _sync_pinecone_query(vec: Any, index_name: str, namespace: Optional[str], top_k: int) -> Any:
    """
    Synchronously query Pinecone with the given vector.
    Returns a list of matches or an error dict.
    """
    try:
        index = pinecone_client.Index(index_name)
        resp = index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        return resp.get("matches", [])
    except Exception as e:
        logging.error(f"Pinecone query failed: {e}")
        return [{"error": f"Pinecone query failed: {e}"}]

