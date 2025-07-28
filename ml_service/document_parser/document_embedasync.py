import logging
import asyncio
from typing import List, Dict, Any
from openai import APIError, RateLimitError, APITimeoutError
from clients import openai_async_client

EMBED_BATCH_SIZE = 100 # <-- Reduced batch size

async def generate_embeddings_async(chunks_with_metadata: list) -> list:
    """
    Generates embeddings for chunks asynchronously, batching requests for efficiency and reliability.
    """
    logging.info(f"Generating embeddings for {len(chunks_with_metadata)} chunks...")

    async def embed_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
            
            # Only retry on specific, transient errors
            except (RateLimitError, APITimeoutError) as e:
                logging.warning(f"Embedding failed on attempt {attempt}, retrying... Error: {e}")
                await asyncio.sleep(2 * attempt) # Exponential backoff
            
            # Do not retry on other errors (like invalid request due to token limit)
            except APIError as e:
                 logging.error(f"Non-retryable OpenAI API error: {e}")
                 # Option 1: Re-raise the exception to stop the process
                 raise e 
                 # Option 2 (if you want to continue but be aware): return [] and handle it later

        # If all retries fail, raise an error instead of failing silently
        raise ConnectionError("Failed to embed batch after multiple retries.")

    # Create tasks for each batch
    batches = [chunks_with_metadata[i:i+EMBED_BATCH_SIZE] for i in range(0, len(chunks_with_metadata), EMBED_BATCH_SIZE)]
    tasks = [embed_batch(batch) for batch in batches]
    
    # Gather results
    results = await asyncio.gather(*tasks)
    
    # Flatten the list of lists
    return [chunk for batch in results for chunk in batch]