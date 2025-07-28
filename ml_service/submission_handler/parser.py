# ml_service/submission_handler/parser.py

import logging
import time
from typing import List, Dict, Any
import asyncio

# Assuming these are the parallel versions we've already refined
from document_parser.document_ingestion import ingest_documents_parallel
from document_parser.document_chunks import chunk_documents_parallel # Using the best version

async def parse_and_chunk_async(specific_file: str) -> List[Dict[str, Any]]:
    """
    Asynchronously runs document ingestion and chunking in parallel without blocking.
    
    Args:
        specific_file: Path to the file to ingest and chunk.
    
    Returns:
        A list of chunk dicts, each containing 'chunk_text' and metadata.
    """
    # 1) Ingest documents in parallel
    ingest_start = time.time()
    documents = await ingest_documents_parallel(specific_file=specific_file)
    ingest_end = time.time()
    logging.info(f"[Timing] Ingested {len(documents)} document(s) in {ingest_end - ingest_start:.2f}s")

    if not documents:
        return []

    # 2) Chunk documents in parallel
    chunk_start = time.time()
    
    # --- THIS IS THE FIX ---
    # Run the synchronous, CPU-bound chunking function in a separate thread
    # to avoid blocking the event loop.
    chunks = await asyncio.to_thread(
        chunk_documents_parallel, documents
    )
    # -----------------------

    chunk_end = time.time()
    logging.info(f"[Timing] Created {len(chunks)} chunk(s) in {chunk_end - chunk_start:.2f}s")

    return chunks
