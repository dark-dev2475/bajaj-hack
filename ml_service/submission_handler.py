# ml_service/submission_handler.py

import os
import logging
import requests
import uuid
import asyncio
from typing import List

import document_parser
import search
from answer import answer_generator

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
    namespace_id = f"{UPLOAD_NAMESPACE_PREFIX}-{uuid.uuid4().hex}"
    save_path = None

    # 1. Download & Ingest
    try:
        resp = requests.get(doc_url, timeout=10)
        resp.raise_for_status()
        filename = doc_url.split("/")[-1].split("?")[0] or f"{namespace_id}.tmp"
        save_path = os.path.join(temp_dir, filename)
        with open(save_path, "wb") as f:
            f.write(resp.content)

        docs   = document_parser.ingest_documents(specific_file=save_path)
        chunks = document_parser.chunk_documents(docs)
        embedded_chunks = await document_parser.generate_embeddings_async(chunks)

        logging.info(f"Upserting {len(embedded_chunks)} vectors to Pinecone namespace '{namespace_id}'…")
        index = pinecone_client.Index(INDEX_NAME)
        # upsert synchronously; if your client supports async, add async_req=True and await
        index.upsert(
            vectors=[
                {
                    "id": f"{namespace_id}-chunk-{i}",
                    "values": c["embedding"],
                    "metadata": {"text": c["chunk_text"], "source": filename}
                }
                for i, c in enumerate(embedded_chunks)
            ],
            namespace=namespace_id
        )

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
            hits = await search.perform_search_async(
                raw_query=q,
                index_name=INDEX_NAME,
                namespace=namespace_id,
                pinecone_client=pinecone_client,
                openai_client=openai_async_client
            )
            if not hits:
                return "No relevant information found."

            ans = await answer_generator.generate_answer_async(
                raw_query=q,
                search_results=hits,
                query_language="en",
                openai_client=openai_async_client
            )
            return ans.Reasoning if ans else "Could not determine an answer."
        except Exception as e:
            logging.error(f"[Answer Error] '{q}': {e}")
            return "Error processing this question."

    answers = await asyncio.gather(*[_answer(q) for q in questions])

    # 3. Cleanup namespace
    try:
        index = pinecone_client.Index(INDEX_NAME)
        index.delete(delete_all=True, namespace=namespace_id)
        logging.info(f"Cleaned up Pinecone namespace '{namespace_id}'.")
    except Exception as e:
        logging.warning(f"[Cleanup Warning] {e}")

    return answers
