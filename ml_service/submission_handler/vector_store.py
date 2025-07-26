# ml_service/submission_handler/vector_store.py

import logging
import time
from typing import List, Dict, Any

from clients import pinecone_client

def upsert_batches(
    vectors: List[Dict[str, Any]],
    index_name: str,
    namespace: str,
    batch_size: int = 500
) -> None:
    """
    Upserts vectors to Pinecone in batches, logging timing.

    Args:
        vectors: List of dicts with 'id', 'values', and 'metadata'.
        index_name: Name of the Pinecone index.
        namespace: Namespace to upsert into.
        batch_size: Number of vectors per upsert batch.
    """
    index = pinecone_client.Index(index_name)
    total_start = time.time()
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        start = time.time()
        index.upsert(vectors=batch, namespace=namespace)
        elapsed = time.time() - start
        logging.info(f"[Timing] Upserted batch {i//batch_size + 1} of {len(batch)} vectors in {elapsed:.2f}s")
    total_elapsed = time.time() - total_start
    logging.info(f"[Timing] Completed upsert of {len(vectors)} vectors in {total_elapsed:.2f}s")


def delete_namespace(
    index_name: str,
    namespace: str
) -> None:
    """
    Deletes all vectors in a Pinecone namespace, logging timing.

    Args:
        index_name: Name of the Pinecone index.
        namespace: Namespace to delete.
    """
    index = pinecone_client.Index(index_name)
    start = time.time()
    index.delete(delete_all=True, namespace=namespace)
    elapsed = time.time() - start
    logging.info(f"[Timing] Deleted namespace '{namespace}' in {elapsed:.2f}s")
