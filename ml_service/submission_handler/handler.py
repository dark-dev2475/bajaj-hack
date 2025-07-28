import os
import logging
import time
from typing import List

from submission_handler.document_loader import async_download_file
from submission_handler.parser import parse_and_chunk
from submission_handler.embedder import embed_chunks
from submission_handler.vector_store import async_upsert_batches, async_delete_namespace
from submission_handler.answering import generate_answers

async def handle_submission(
    doc_url: str,
    questions: List[str],
    temp_dir: str,
    index_name: str,
    namespace_prefix: str = "rag",
    batch_size: int = 500
) -> List[str]:
    """
    High-level orchestration of the RAG pipeline.
    """
    start_total = time.time()
    namespace_id = f"{namespace_prefix}-{int(start_total)}"
    logging.info(f"Starting new RAG pipeline with namespace_id: {namespace_id}")

    save_path, filename = await async_download_file(doc_url, temp_dir)
    if not save_path:
        # Handle download failure
        return ["Failed to download document."] * len(questions)

    try:
        # 1) Parse & chunk
        chunks = await parse_and_chunk(save_path)
        logging.info(f"[{namespace_id}] Created {len(chunks)} chunks from document '{filename}'.")
        if not chunks:
            # Handle case where no text could be extracted
            return ["Could not parse any text from the document."] * len(questions)

        # 2) Embed in batches
        embedded_chunks = await embed_chunks(chunks, batch_size=batch_size)
        logging.info(f"[{namespace_id}] Generated {len(embedded_chunks)} embeddings.")

        # Prepare vector dicts
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

        # 3) Upsert to Pinecone
        await async_upsert_batches(vectors, index_name, namespace_id, batch_size=batch_size)

        # 4) Generate answers
        answers = await generate_answers(questions, namespace_id, index_name)
        return answers

    except Exception as e:
        # CRITICAL: Use logging.exception to get the full stack trace in your logs
        logging.exception(f"[{namespace_id}] A critical error occurred in the RAG pipeline.")
        return [f"Pipeline failed due to an internal error."] * len(questions)

    finally:
        # Cleanup
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