# ml_service/submission_handler.py

import os
import logging
import requests
import uuid
import asyncio
from typing import List
import time

from document_parser import ingest_documents, chunk_documents, generate_embeddings_async
import search
from answer import answer_generator
from query_parser.main_parser import get_structured_query

# These clients must be initialized in clients.py:
#   pinecone_client = pinecone.init(...)
#   openai_async_client = AsyncOpenAI(...)
from clients import pinecone_client, openai_async_client  

INDEX_NAME = "polisy-search"      # <-- your actual Pinecone index name
UPLOAD_NAMESPACE_PREFIX = "rag"  # prefix for clarity, optional

async def handle_submission(doc_url: str, questions: List[str], temp_dir: str) -> List[str]:
    """
    Orchestrates the RAG pipeline:
     1) download & parse
     2) chunk & embed & index
     3) retrieve & generate answers
     4) cleanup
    """
    pipeline_start = time.time()
    namespace_id = f"{UPLOAD_NAMESPACE_PREFIX}-{uuid.uuid4().hex}"
    save_path = None

    # 1. Download & Ingest
    try:
        t0 = time.time()
        resp = requests.get(doc_url, timeout=10)
        resp.raise_for_status()
        filename = doc_url.split("/")[-1].split("?")[0] or f"{namespace_id}.tmp"
        save_path = os.path.join(temp_dir, filename)
        with open(save_path, "wb") as f:
            f.write(resp.content)
        t1 = time.time()
        logging.info(f"[Timing] Document download: {t1-t0:.2f}s")

        t0 = time.time()
        docs   = ingest_documents(specific_file=save_path)
        chunks = chunk_documents(docs)
        t1 = time.time()
        logging.info(f"[Timing] Ingest + chunk: {t1-t0:.2f}s (chunks: {len(chunks)})")

        # Pipeline embedding and upsert for maximum throughput
        BATCH_SIZE = 500  # Batch size for both embedding and upsert
        vectors = []
        upsert_tasks = []
        index = pinecone_client.Index(INDEX_NAME)
        async def upsert_batch(batch):
            start = time.time()
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: index.upsert(vectors=batch, namespace=namespace_id)
            )
            end = time.time()
            logging.info(f"[Timing] Upserted batch of {len(batch)} in {end-start:.2f}s")
            return result
        # Process in batches: embed, then immediately upsert while next batch is embedding
        embed_total = 0
        for i in range(0, len(chunks), BATCH_SIZE):
            chunk_batch = chunks[i:i+BATCH_SIZE]
            embed_start = time.time()
            embedded_batch = await generate_embeddings_async(chunk_batch)
            embed_end = time.time()
            embed_total += (embed_end - embed_start)
            logging.info(f"[Timing] Embedded batch of {len(chunk_batch)} in {embed_end-embed_start:.2f}s")
            batch_vectors = [
                {
                    "id": f"{namespace_id}-chunk-{i+j}",
                    "values": c["embedding"],
                    "metadata": {"text": c["chunk_text"], "source": filename}
                }
                for j, c in enumerate(embedded_batch)
            ]
            # Start upsert for this batch while next batch is embedding
            upsert_tasks.append(asyncio.create_task(upsert_batch(batch_vectors)))
            vectors.extend(batch_vectors)
        # Wait for all upserts to finish
        upsert_start = time.time()
        await asyncio.gather(*upsert_tasks)
        upsert_end = time.time()
        logging.info(f"[Timing] Total embedding time: {embed_total:.2f}s")
        logging.info(f"[Timing] Total upsert time: {upsert_end-upsert_start:.2f}s")

    except Exception as e:
        logging.error(f"[Ingestion Error] {e}")
        # On failure, return error strings for each question
        return [f"Failed to ingest document: {e}"] * len(questions)

    finally:
        if save_path and os.path.exists(save_path):
            os.remove(save_path)

    # 2. Retrieve & Generate (parallel)
    async def _answer(q: str) -> str:
        try:
            answer_start = time.time()
            structured_query = get_structured_query(q)
            hits = await search.perform_search_async(
                raw_query=q,
                index_name=INDEX_NAME,
                namespace=namespace_id,
                pinecone_client=pinecone_client,
                openai_client=openai_async_client,
                structured_query=structured_query
            )
            if not hits:
                return "No relevant information found."

            ans = await answer_generator.generate_answer_async(
                raw_query=q,
                search_results=hits,
                query_language="en",
                openai_client=openai_async_client
            )
            answer_end = time.time()
            logging.info(f"[Timing] Answer generation for query '{q[:30]}...': {answer_end-answer_start:.2f}s")
            return ans.Reasoning if ans else "Could not determine an answer."
        except Exception as e:
            logging.error(f"[Answer Error] '{q}': {e}")
            return "Error processing this question."

    answers_start = time.time()
    answers = await asyncio.gather(*[_answer(q) for q in questions])
    answers_end = time.time()
    logging.info(f"[Timing] All answers generated in {answers_end-answers_start:.2f}s for {len(questions)} queries")

    # 3. Cleanup namespace
    try:
        cleanup_start = time.time()
        index = pinecone_client.Index(INDEX_NAME)
        index.delete(delete_all=True, namespace=namespace_id)
        cleanup_end = time.time()
        logging.info(f"[Timing] Cleanup time: {cleanup_end-cleanup_start:.2f}s")
        logging.info(f"[Timing] Total pipeline time: {cleanup_end-pipeline_start:.2f}s")
    except Exception as e:
        logging.warning(f"[Cleanup Warning] {e}")

    return answers