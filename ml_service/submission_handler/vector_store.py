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
    Asynchronously and concurrently upserts vectors to Pinecone in batches.

    Args:
        vectors: List of dicts with 'id', 'values', and 'metadata'.
        index_name: Name of the Pinecone index.
        namespace: Namespace to upsert into.
        batch_size: Number of vectors per upsert batch.
    """
    index = pinecone_client.Index(index_name)
    total_start = time.time()
    logging.info(f"Starting upsert of {len(vectors)} vectors to namespace '{namespace}'...")

    # Create tasks for each batch using the modern asyncio.to_thread
    tasks = []
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i: i + batch_size]
        task = asyncio.to_thread(index.upsert, vectors=batch, namespace=namespace)
        tasks.append(task)
    
    # Use return_exceptions=True to prevent one failure from stopping all others
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check results for any exceptions that occurred
    success_count = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logging.error(f"Batch {i+1} failed to upsert: {result}")
        else:
            success_count += 1
    
    logging.info(
        f"Upsert complete for namespace '{namespace}': "
        f"{success_count}/{len(tasks)} batches succeeded in {time.time() - total_start:.2f}s"
    )

async def async_delete_namespace(
    index_name: str,
    namespace: str
) -> None:
    """
    Asynchronously deletes all vectors in a Pinecone namespace.

    Args:
        index_name: Name of the Pinecone index.
        namespace: Namespace to delete.
    """
    index = pinecone_client.Index(index_name)
    start = time.time()
    logging.info(f"Attempting to delete namespace '{namespace}'...")

    try:
        # Simply use asyncio.to_thread directly on the method
        await asyncio.to_thread(index.delete, delete_all=True, namespace=namespace)
        elapsed = time.time() - start
        logging.info(f"Successfully deleted namespace '{namespace}' in {elapsed:.2f}s")
    except Exception as e:
        logging.error(f"Failed to delete namespace '{namespace}': {e}")