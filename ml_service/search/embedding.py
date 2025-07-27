import logging
from clients import openai_async_client,openai_client
from typing import Any

async def _async_embed_query(query: str, openai_client: openai_async_client) -> Any:
    """
    Asynchronously embed a query string using the provided OpenAI client.
    Returns the embedding vector or an error dict.
    """
    try:
        resp = await openai_client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        )
        return resp.data[0].embedding
    except Exception as e:
        logging.error(f"[Async] OpenAI embedding failed: {e}")
        return {"error": f"OpenAI embedding failed: {e}"}


# def sync_embed_query(query: str, openai_client: openai_client):
#     try:
#         return openai_client.embeddings.create(
#             input=[query],
#             model="text-embedding-3-small"
#         ).data[0].embedding
#     except Exception as e:
#         logging.error(f"OpenAI embedding failed: {e}")
#         return {"error": f"OpenAI embedding failed: {e}"}



