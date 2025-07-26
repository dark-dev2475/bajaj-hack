# ml_service/submission_handler/embedder.py

import logging
import time
import asyncio
from typing import List, Dict, Any

from document_parser.document_embedasync import generate_embeddings_async

async def embed_chunks(
    chunks: List[Dict[str, Any]],
    batch_size: int = 500
) -> List[Dict[str, Any]]:
    """
    Generates embeddings for the given chunks in batches, logging timing.

    Args:
        chunks: List of chunk dicts (with 'chunk_text').
        batch_size: Number of chunks per batch.

    Returns:
        The same list of chunk dicts, each augmented with an 'embedding' field.
    """
    total_start = time.time()
    embedded_chunks: List[Dict[str, Any]] = []

    # Process in sub‐batches to control memory and API load
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        start = time.time()
        # generate_embeddings_async itself handles retries and returns the batch
        enriched_batch = await generate_embeddings_async(batch)
        elapsed = time.time() - start
        logging.info(f"[Timing] Embedded batch {i//batch_size + 1} of {len(batch)} chunks in {elapsed:.2f}s")
        embedded_chunks.extend(enriched_batch)

    total_elapsed = time.time() - total_start
    logging.info(f"[Timing] Total embedding time for {len(chunks)} chunks: {total_elapsed:.2f}s")
    return embedded_chunks
