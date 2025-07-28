# ml_service/submission_handler/vector_store.py

import logging
import time
from typing import List, Dict, Any
import asyncio
from clients import pinecone_client
from pinecone.core.client.exceptions import ApiException # For specific error handling

async def async_upsert_batches(
    vectors: List[Dict[str, Any]],
    index_name: str,
    namespace: str,
    batch_size: int = 50, # Smaller batches are often more reliable in production
    concurrency_limit: int = 10,
    max_retries: int = 3
) -> None:
    """
    Asynchronously upserts vectors to Pinecone with controlled concurrency, retries,
    and response verification to ensure data is successfully indexed.
    """
    index = pinecone_client.Index(index_name)
    total_start = time.time()
    logging.info(f"Starting upsert of {len(vectors)} vectors to namespace '{namespace}'...")

    semaphore = asyncio.Semaphore(concurrency_limit)

    async def upsert_with_retry(batch: List[Dict[str, Any]], batch_num: int):
        """
        A wrapper function that attempts to upsert a batch with retries and response verification.
        """
        async with semaphore:
            for attempt in range(1, max_retries + 1):
                try:
                    # Capture the response from the upsert call
                    response = await asyncio.to_thread(
                        index.upsert, vectors=batch, namespace=namespace
                    )
                    
                    # --- CRITICAL IMPROVEMENT: Verify the response ---
                    # Check if the number of upserted vectors matches the batch size.
                    if response.upserted_count != len(batch):
                        # This catches silent failures where no exception is thrown.
                        raise ValueError(
                            f"Upsert count mismatch. Sent {len(batch)}, "
                            f"but Pinecone reported {response.upserted_count} upserted."
                        )
                    # --- END OF IMPROVEMENT ---

                    logging.info(f"Successfully upserted and verified batch {batch_num} of {len(batch)} vectors.")
                    return True # Indicate success

                except ApiException as e:
                    if e.status >= 500 and attempt < max_retries:
                        logging.warning(
                            f"Batch {batch_num} failed with server error (status {e.status}). "
                            f"Retrying in {2**attempt}s... (Attempt {attempt}/{max_retries})"
                        )
                        await asyncio.sleep(2**attempt)
                    else:
                        logging.error(f"Batch {batch_num} failed with non-retryable API error: {e}")
                        return e
                except Exception as e:
                    # This will now catch our custom ValueError as well as other issues.
                    logging.error(f"An unexpected error occurred upserting batch {batch_num} on attempt {attempt}: {e}")
                    if attempt < max_retries:
                        await asyncio.sleep(2**attempt)
                    else:
                        return e

            logging.error(f"Batch {batch_num} failed after {max_retries} attempts.")
            return False

    # Create tasks for each batch
    tasks = []
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i: i + batch_size]
        tasks.append(upsert_with_retry(batch, (i // batch_size) + 1))
    
    results = await asyncio.gather(*tasks)

    success_count = sum(1 for r in results if r is True)
    logging.info(
        f"Upsert process finished for namespace '{namespace}': "
        f"{success_count}/{len(tasks)} batches succeeded in {time.time() - total_start:.2f}s"
    )

async def async_delete_namespace(
    index_name: str,
    namespace: str
) -> None:
    """
    Asynchronously deletes all vectors in a Pinecone namespace.
    """
    index = pinecone_client.Index(index_name)
    start = time.time()
    logging.info(f"Attempting to delete namespace '{namespace}'...")

    try:
        await asyncio.to_thread(index.delete, delete_all=True, namespace=namespace)
        elapsed = time.time() - start
        logging.info(f"Successfully deleted namespace '{namespace}' in {elapsed:.2f}s")
    except Exception as e:
        logging.error(f"Failed to delete namespace '{namespace}': {e}")
