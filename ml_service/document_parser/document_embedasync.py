import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union
from langchain_openai import OpenAIEmbeddings

# Initialize OpenAI embeddings with the most efficient model
from clients import openai_async_client

def normalize_vector(vector: Union[List[float], np.ndarray]) -> np.ndarray:
    """Normalize a vector to unit length using L2 normalization."""
    if isinstance(vector, list):
        vector = np.array(vector)
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector

def normalize_vectors_batch(vectors: List[List[float]]) -> List[np.ndarray]:
    """Normalize a batch of vectors to unit length."""
    return [normalize_vector(vec) for vec in vectors]

# Configuration
BATCH_SIZE = 100  # OpenAI can handle larger batches efficiently
MAX_CONCURRENT_BATCHES = 5  # Limit concurrent API calls to avoid rate limits
MAX_RETRIES = 3

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # Using the latest, most efficient model
    client=openai_async_client,
    chunk_size=BATCH_SIZE
)

async def generate_embeddings_async(chunks_with_metadata: list) -> list:
    """
    Generates embeddings for chunks using OpenAI's text-embedding-3-small model,
    processing batches in parallel with controlled concurrency and retry mechanism.
    
    The text-embedding-3-small model provides:
    - Optimal performance for most use cases
    - Lower latency and cost compared to larger models
    - 1536-dimensional embeddings
    """
    if not chunks_with_metadata:
        return []
        
    logging.info(f"Starting OpenAI embedding process for {len(chunks_with_metadata)} chunks using text-embedding-3-small.")

    async def embed_batch_with_retry(batch: List[Dict[str, Any]], batch_num: int) -> List[Dict[str, Any]]:
        """Processes a single batch with an exponential backoff retry mechanism."""
        texts = [chunk["chunk_text"] for chunk in batch]
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if not openai_async_client:
                    raise ConnectionError("OpenAI client not initialized.")
                
                # Use OpenAI's efficient batch embedding
                embeddings_result = await embeddings.aembed_documents(texts)
                
                # Attach embeddings back to their original chunks
                for i, chunk in enumerate(batch):
                    chunk["embedding"] = embeddings_result[i]
                    chunk["embedding_model"] = "text-embedding-3-small"  # Track which model was used
                
                logging.info(f"Successfully embedded batch {batch_num} ({len(batch)} chunks) with OpenAI.")
                return batch # Success
            
            except Exception as e:
                logging.warning(f"Embedding batch {batch_num} failed on attempt {attempt}/{MAX_RETRIES}: {e}")
                if attempt == MAX_RETRIES:
                    logging.error(f"All embedding attempts failed for batch {batch_num}. The batch will be skipped.")
                    return [] # Return empty list for the failed batch
                await asyncio.sleep(2 ** attempt) # Exponential backoff
        return [] # Should not be reached, but as a safeguard

    # --- Concurrency Control ---
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)
    batches = [chunks_with_metadata[i:i+BATCH_SIZE] for i in range(0, len(chunks_with_metadata), BATCH_SIZE)]
    
    async def process_with_semaphore(batch: List[Dict[str, Any]], batch_num: int):
        async with semaphore:
            return await embed_batch_with_retry(batch, batch_num)

    tasks = [process_with_semaphore(batch, i+1) for i, batch in enumerate(batches)]
    
    try:
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten the list of lists into a single list of successfully processed chunks
        processed_chunks = [chunk for batch in batch_results for chunk in batch]
        
        if len(processed_chunks) < len(chunks_with_metadata):
            logging.warning(f"Embedding completed, but {len(chunks_with_metadata) - len(processed_chunks)} chunks were lost due to errors.")
        else:
            logging.info("All chunks embedded successfully.")
            
        return processed_chunks
        
    except Exception as e:
        logging.exception(f"A critical, unhandled error occurred during the embedding process: {e}")
        return []
