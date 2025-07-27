# ml_service/submission_handler/vector_store.py

import logging
import time
from typing import List, Dict, Any
import asyncio
from clients import pinecone_client

async def async_upsert_batches(
    vectors: List[Dict[str, Any]],
    index_name: str,
    namespace: str,
    batch_size: int = 500
) -> None:
    """
    Asynchronously upserts vectors to Pinecone in batches.

    Args:
        vectors: List of dicts with 'id', 'values', and 'metadata'.
        index_name: Name of the Pinecone index.
        namespace: Namespace to upsert into.
        batch_size: Number of vectors per upsert batch.
    """
    index = pinecone_client.Index(index_name)
    total_start = time.time()

    async def upsert_batch(batch, batch_num):
        loop = asyncio.get_event_loop()
        start = time.time()
        await loop.run_in_executor(None, index.upsert, batch, namespace)
        elapsed = time.time() - start
        logging.info(f"[Timing] Upserted batch {batch_num} of {len(batch)} vectors in {elapsed:.2f}s")

    tasks = []
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i: i + batch_size]
        tasks.append(upsert_batch(batch, i // batch_size + 1))

    await asyncio.gather(*tasks)
    logging.info(f"[Timing] Completed upsert of {len(vectors)} vectors in {time.time() - total_start:.2f}s")

async def async_delete_namespace(
    index_name: str,
    namespace: str
) -> None:
    """
    Asynchronously deletes all vectors in a Pinecone namespace, logging timing.

    Args:
        index_name: Name of the Pinecone index.
        namespace: Namespace to delete.
    """
    loop = asyncio.get_event_loop()
    index = pinecone_client.Index(index_name)

    start = time.time()

    def blocking_delete():
        index.delete(delete_all=True, namespace=namespace)

    await loop.run_in_executor(None, blocking_delete)

    elapsed = time.time() - start
    logging.info(f"[Timing] Deleted namespace '{namespace}' in {elapsed:.2f}s")

