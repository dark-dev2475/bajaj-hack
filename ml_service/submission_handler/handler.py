# ml_service/submission_handler/handler.py

import os
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional

# Import the new, improved, and parallel versions of our functions
from .document_loader import async_download_file
from .parser import parse_and_chunk_async
from .embedder import embed_chunks
from .vector_store import async_upsert_batches, async_delete_namespace
from .answering import generate_answers
# We need an extractor to get the policy type from the document
from query_parser.llm_extractor import extract_with_llm_async

async def _extract_policy_type_from_doc(text: str) -> Optional[str]:
    """
    Uses OpenAI to extract the policy type from the document.
    Optimized for accurate policy type identification.
    """
    # Take a larger context for better accuracy but not too large
    snippet = text[:2000]  # Increased context window
    logging.info("Extracting policy_type using OpenAI...")
    try:
        # Using a more specific prompt for better accuracy
        structured_query = (
            "Analyze this insurance document and determine the exact policy type. "
            "Common types include: health, life, motor, property, travel, or liability. "
            "Focus on explicit policy type mentions in headers or opening paragraphs.\n\n"
            f"Document start: {snippet}"
        )
        extracted_data = await extract_with_llm_async(structured_query)
        
        if extracted_data and extracted_data.get("policy_type"):
            policy_type = extracted_data["policy_type"]
            # Normalize and validate the policy type
            normalized_policy_type = policy_type.lower().strip()
            # Ensure we have a valid policy type
            valid_types = {"health", "life", "motor", "property", "travel", "liability"}
            if any(valid_type in normalized_policy_type for valid_type in valid_types):
                logging.info(f"Successfully extracted policy type: '{normalized_policy_type}'")
                return normalized_policy_type
            else:
                logging.warning(f"Extracted policy type '{normalized_policy_type}' not in standard categories")
                return "other"
    except Exception as e:
        logging.error(f"Failed to extract policy_type using OpenAI: {e}")
        # Return a default value for graceful handling
        return "unspecified"


async def _ingest_document(
    file_path: str,
    filename: str,
    namespace_id: str,
    index_name: str
) -> bool:
    """
    A helper function to encapsulate the document ingestion process.
    1. Parse & Chunk
    2. Enrich Metadata with policy_type
    3. Embed Chunks
    4. Upsert to Vector Store
    """
    logging.info(f"[{namespace_id}] Starting ingestion for '{filename}'...")
    
    chunks = await parse_and_chunk_async(file_path)
    if not chunks:
        logging.error(f"[{namespace_id}] No chunks were created from the document. Aborting.")
        return False

    # --- METADATA ENRICHMENT ---
    # Extract the policy type once from the first chunk
    policy_type = await _extract_policy_type_from_doc(chunks[0]['chunk_text'])

    embedded_chunks = await embed_chunks(chunks)
    if not embedded_chunks:
        logging.error(f"[{namespace_id}] Failed to generate embeddings for chunks. Aborting.")
        return False

    vectors = [
        {
            "id": f"{namespace_id}-chunk-{i}",
            "values": chunk["embedding"],
            "metadata": {
                "text": chunk["chunk_text"],
                "source": filename,
                # Add the extracted policy_type to every chunk's metadata
                "policy_type": policy_type or "unknown" 
            }
        }
        for i, chunk in enumerate(embedded_chunks)
    ]
    logging.info(f"[{namespace_id}] Prepared {len(vectors)} vectors for upsert with enriched metadata.")

    await async_upsert_batches(vectors, index_name, namespace_id)
    
    logging.info(f"[{namespace_id}] Waiting for indexing to complete...")
    await asyncio.sleep(10) # Increased delay for better indexing consistency
    
    return True


async def handle_submission(
    doc_url: str,
    questions: List[str],
    temp_dir: str,
    index_name: str,
) -> List[Dict[str, Any]]:
    """
    High-level orchestration of the RAG pipeline.
    """
    start_total = time.time()
    namespace_id = f"rag-{int(start_total)}"
    logging.info(f"Starting new RAG pipeline with namespace_id: {namespace_id}")

    save_path, filename = await async_download_file(doc_url, temp_dir)
    if not save_path:
        return [{"error": "Failed to download document."}] * len(questions)

    try:
        ingestion_success = await _ingest_document(save_path, filename, namespace_id, index_name)
        if not ingestion_success:
            return [{"error": "Failed to process and ingest document."}] * len(questions)

        answers = await generate_answers(questions, namespace_id, index_name)

        results = [ans.model_dump() if ans else {"error": "Failed to generate answer."} for ans in answers]
        return results

    except Exception as e:
        logging.exception(f"[{namespace_id}] A critical unhandled error occurred in the main pipeline.")
        return [{"error": "A critical internal error occurred."}] * len(questions)

    finally:
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