import logging
import asyncio

# --- Embedding: Async ---
EMBED_BATCH_SIZE = 500

async def generate_embeddings_async(chunks_with_metadata: list) -> list:
    """
    Generates embeddings for chunks asynchronously, batching requests for efficiency and reliability.
    Batches are processed in parallel and retried up to 3 times on failure.
    """
    logging.info("Generating embeddings (async, batched)...")
    async def embed_batch(batch):
        texts = [chunk["chunk_text"] for chunk in batch]
        for _ in range(3):  # Retry up to 3 times
            try:
                response = await async_client.embeddings.create(
                    input=texts,
                    model="text-embedding-3-small"
                )
                for i, chunk in enumerate(batch):
                    chunk["embedding"] = response.data[i].embedding
                return batch
            except Exception as e:
                logging.error(f"Async embedding failed for batch: {e}")
                await asyncio.sleep(1)
        return []

    batches = [chunks_with_metadata[i:i+EMBED_BATCH_SIZE] for i in range(0, len(chunks_with_metadata), EMBED_BATCH_SIZE)]
    results = await asyncio.gather(*(embed_batch(batch) for batch in batches))
    # Flatten the list of lists
    return [chunk for batch in results for chunk in batch]