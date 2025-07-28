# ml_service/submission_handler/parser.py

import logging
import time
from typing import List, Dict, Any
import asyncio

from document_parser.document_ingestion import ingest_documents_parallel
from document_parser.document_chunks import  chunk_documents_parallel

async def parse_and_chunk(specific_file: str) -> List[Dict[str, Any]]:
    """
    Runs document ingestion and chunking, logging the time taken for each step.
    
    Args:
        specific_file: Path to the file to ingest and chunk.
    
    Returns:
        A list of chunk dicts, each containing 'chunk_text' and metadata.
    """
    # 1) Ingest
    t0 = time.time()
    documents = await ingest_documents_parallel(specific_file=specific_file)
    t1 = time.time()
    logging.info(f"[Timing] Ingested {len(documents)} document(s) in {t1 - t0:.2f}s")

    # 2) Chunk
    t2 = time.time()
    chunks = chunk_documents_parallel(documents)
    t3 = time.time()
    logging.info(f"[Timing] Created {len(chunks)} chunk(s) in {t3 - t2:.2f}s")

    return chunks




