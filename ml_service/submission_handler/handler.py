# ml_service/submission_handler/handler.py

import os
import logging
import time
import asyncio
from typing import List, Dict, Any

# Import the new, improved, and parallel versions of our functions
from .document_loader import async_download_file
from .parser import parse_and_chunk_async # Using the async, non-blocking version
from .embedder import embed_chunks_parallel # Using the parallel version
from .vector_store import async_upsert_batches, async_delete_namespace
from .answering import generate_answers_in_parallel # This is our final, parallel answering function

async def _ingest_document(
    file_path: str,
    filename: str,
    namespace_id: str,
    index_name: str
) -> bool:
    """
    A helper function to encapsulate the document ingestion process.
    1. Parse & Chunk
    2. Embed Chunks
    3. Prepare Vectors
    4. Upsert to Vector Store
    """
    logging.info(f"[{namespace_id}] Starting ingestion for '{filename}'...")
    
    # 1) Parse & chunk using our improved async parser
    chunks = await parse_and_chunk_async(file_path)
    if not chunks:
        logging.error(f"[{namespace_id}] No chunks were created from the document. Aborting.")
        return False

    # 2) Embed chunks in parallel
    embedded_chunks = await embed_chunks_parallel(chunks)
    if not embedded_chunks:
        logging.error(f"[{namespace_id}] Failed to generate embeddings for chunks. Aborting.")
        return False

    # 3) Prepare vector dictionaries for Pinecone
    vectors = [
        {
            "id": f"{namespace_id}-chunk-{i}",
            "values": chunk["embedding"],
            "metadata": {
                "text": chunk["chunk_text"],
                "source": filename
            }
        }
        for i, chunk in enumerate(embedded_chunks)
    ]
    logging.info(f"[{namespace_id}] Prepared {len(vectors)} vectors for upsert.")

    # 4) Upsert to Pinecone using our robust, parallel upsert function
    await async_upsert_batches(vectors, index_name, namespace_id)
    
    # Give the vector database a moment to index the new data
    logging.info(f"[{namespace_id}] Waiting for indexing to complete...")
    await asyncio.sleep(5)
    
    return True


async def handle_submission(
    doc_url: str,
    questions: List[str],
    temp_dir: str,
    index_name: str,
    namespace_prefix: str = "rag"
) -> List[Dict[str, Any]]:
    """
    High-level orchestration of the RAG pipeline.
    """
    start_total = time.time()
    namespace_id = f"{namespace_prefix}-{int(start_total)}"
    logging.info(f"Starting new RAG pipeline with namespace_id: {namespace_id}")

    save_path, filename = await async_download_file(doc_url, temp_dir)
    if not save_path:
        return [{"error": "Failed to download document."}] * len(questions)

    try:
        # --- Step 1: Ingest the document ---
        ingestion_success = await _ingest_document(save_path, filename, namespace_id, index_name)
        if not ingestion_success:
            return [{"error": "Failed to process and ingest document."}] * len(questions)

        # --- Step 2: Generate answers in parallel ---
        # This now calls our final, improved answering function.
        answers = await generate_answers_in_parallel(questions, namespace_id, index_name)
        
        # --- Step 3: Convert results to JSON-serializable format ---
        # The pipeline now returns structured objects. We convert them to dicts for the API.
        results = [ans.model_dump() if ans else {"error": "Failed to generate answer."} for ans in answers]
        return results

    except Exception as e:
        logging.exception(f"[{namespace_id}] A critical unhandled error occurred in the main pipeline.")
        return [{"error": "A critical internal error occurred."}] * len(questions)

    finally:
        # --- Step 4: Cleanup ---
        logging.info(f"[{namespace_id}] Starting cleanup process.")
        try:
            await async_delete_namespace(index_name, namespace_id)
            logging.info(f"[{namespace_id}] Successfully deleted namespace.")
        except Exception as cleanup_err:
            logging.warning(f"[{namespace_id}] Could not delete namespace during cleanup: {cleanup_err}")

        if os.path.exists(save_path):
            os.remove(save_path)
            logging.info(f"[{namespace_id}] Successfully removed temporary file '{save_path}'.")

        logging.info(f"[{namespace_id}] Total pipeline time: {time.time() - start_total:.2f}s")
