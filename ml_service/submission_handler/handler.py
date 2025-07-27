# ml_service/submission_handler/handler.py

import os
import logging
import time
from typing import List

from submission_handler.document_loader import async_download_file
from submission_handler.parser import parse_and_chunk_async
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
    High‑level orchestration of the RAG pipeline:
      1) Download
      2) Parse + Chunk
      3) Embed + Upsert
      4) Answer Generation
      5) Cleanup

    Returns:
        List of answers corresponding to the input questions.
    """
    start_total = time.time()
    # Use timestamp instead of uuid for namespace
    namespace_id = f"{namespace_prefix}-{int(start_total)}"

    # 1) Download
    download_start = time.time()
    save_path, filename = await async_download_file(doc_url, temp_dir)  
    logging.info(f"[Timing] Total download step: {time.time() - download_start:.2f}s")

    try:
        # 2) Parse & chunk
        chunks = await parse_and_chunk_async(save_path)

        # 3) Embed in batches
        embedded_chunks = await embed_chunks(chunks, batch_size=batch_size)

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

        # 4) Upsert to Pinecone
        upsert_start = time.time()
        async_upsert_batches(vectors, index_name, namespace_id, batch_size=batch_size)
        logging.info(f"[Timing] Upsert step: {time.time() - upsert_start:.2f}s")

        # 5) Generate answers
        answers_start = time.time()
        answers = await generate_answers(questions, namespace_id, index_name)
        logging.info(f"[Timing] Answer generation step: {time.time() - answers_start:.2f}s")
        return answers

    except Exception as e:
        logging.error(f"[Pipeline Error] {e}")
        # Return error per question
        return [f"Pipeline failed: {e}"] * len(questions)

    finally:
        # Cleanup namespace
        try:
            async_delete_namespace(index_name, namespace_id)
        except Exception as cleanup_err:
            logging.warning(f"[Cleanup Warning] {cleanup_err}")

        # Remove downloaded file
        if os.path.exists(save_path):
            os.remove(save_path)

        logging.info(f"[Timing] Total pipeline time: {time.time() - start_total:.2f}s")
