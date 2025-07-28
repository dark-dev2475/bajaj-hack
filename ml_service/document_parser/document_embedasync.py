import logging
import asyncio
from typing import List, Dict, Any
from openai import APIError, RateLimitError, APITimeoutError
from clients import openai_async_client

# A smaller, safer batch size to avoid exceeding the API's token limit.
EMBED_BATCH_SIZE = 100

async def generate_embeddings_async(chunks_with_metadata: list) -> list:
    """
    Generates embeddings for chunks asynchronously, batching requests for efficiency and reliability.
    """
    if not chunks_with_metadata:
        return []
        
    logging.info(f"Generating embeddings for {len(chunks_with_metadata)} chunks...")

    async def embed_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes a single batch with a smart retry mechanism."""
        texts = [chunk["chunk_text"] for chunk in batch]
        
        for attempt in range(1, 4):  # Retry up to 3 times
            try:
                response = await openai_async_client.embeddings.create(
                    input=texts,
                    model="text-embedding-3-small"
                )
                # Attach embeddings back to their original chunks
                for i, chunk in enumerate(batch):
                    chunk["embedding"] = response.data[i].embedding
                return batch # Success
            
            # Only retry on specific, transient network/API errors
            except (RateLimitError, APITimeoutError) as e:
                logging.warning(f"Embedding failed on attempt {attempt}, retrying... Error: {e}")
                await asyncio.sleep(2 * attempt) # Exponential backoff
            
            # Fail immediately on non-retryable errors (e.g., batch too large)
            except APIError as e:
                 logging.error(f"Non-retryable OpenAI API error for a batch: {e}")
                 # Re-raising the error stops the process, preventing silent data loss.
                 raise e

    # Create tasks for each batch
    batches = [chunks_with_metadata[i:i+EMBED_BATCH_SIZE] for i in range(0, len(chunks_with_metadata), EMBED_BATCH_SIZE)]
    tasks = [embed_batch(batch) for batch in batches]
    
    try:
        # Gather results from all concurrent tasks
        results = await asyncio.gather(*tasks)
        
        # Flatten the list of lists into a single list of chunks
        return [chunk for batch in results for chunk in batch]
    except Exception as e:
        logging.exception("A critical error occurred during the parallel embedding process.")
        # Depending on desired behavior, you could return an empty list or re-raise
        return []

