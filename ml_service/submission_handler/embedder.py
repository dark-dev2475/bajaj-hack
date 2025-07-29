import logging
import time
import asyncio
from typing import List, Dict, Any

from document_parser.document_embedasync import generate_embeddings_async

async def embed_chunks(
    chunks: List[Dict[str, Any]],
    batch_size: int = 100
) -> List[Dict[str, Any]]:
    """
    Generates embeddings for chunks in parallel batches for maximum performance.
    """
    logging.info(f"Starting to embed {len(chunks)} chunks in parallel batches of {batch_size}...")
    total_start = time.time()
    
    # Create a list of tasks, one for each batch
    tasks = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        # Each call to generate_embeddings_async becomes a task
        task = generate_embeddings_async(batch)
        tasks.append(task)
    
    # Run all batch-embedding tasks concurrently
    logging.info(f"Sending {len(tasks)} batches to be processed in parallel.")
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    
    # Combine the results from all batches
    embedded_chunks = [chunk for batch in batch_results for chunk in batch]
    
    total_elapsed = time.time() - total_start
    logging.info(f"Completed embedding {len(embedded_chunks)} out of {len(chunks)} chunks in {total_elapsed:.2f}s")
    
    return embedded_chunks