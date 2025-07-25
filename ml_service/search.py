# ml_service/search.py

import os
import logging
from typing import List, Dict, Any, Optional

import pinecone
from openai import OpenAI
from langdetect import detect
from query_parser.schema import PolicyQuery

# — Sync clients (lazily initialized) —
_sync_openai: Optional[OpenAI] = None
_pinecone_client: Optional[pinecone.Pinecone] = None

def _init_sync_clients():
    global _sync_openai, _pinecone_client
    if not _sync_openai:
        _sync_openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if not _pinecone_client:
        _pinecone_client = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])


# — Async clients: you should import and pass these from your `clients.py` —
# from clients import pinecone_client, openai_async_client


async def perform_search_async(
    raw_query: str,
    index_name: str,
    namespace: str,
    pinecone_client: pinecone.Pinecone,
    openai_client
) -> List[Dict[str, Any]]:
    """
    Asynchronously embeds the query and runs a Pinecone search.
    """
    # 1) Embed query
    try:
        resp = await openai_client.embeddings.create(
            input=[raw_query],
            model="text-embedding-3-small"
        )
        query_vec = resp.data[0].embedding
    except Exception as e:
        logging.error(f"[Async] OpenAI embedding failed: {e}")
        return []

    # 2) Search vector DB
    try:
        index = pinecone_client.Index(index_name)
        response = index.query(
            vector=query_vec,
            top_k=5,
            include_metadata=True,
            namespace=namespace
        )
        return response.get("matches", [])
    except Exception as e:
        logging.error(f"[Async] Pinecone query failed: {e}")
        return []


def perform_search(
    raw_query: str,
    index_name: str,
    namespace: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Synchronously embeds the query and runs a Pinecone search.
    """
    _init_sync_clients()
    if not _sync_openai or not _pinecone_client:
        logging.error("Sync clients not initialized.")
        return []

    # 1) Possibly translate
    query = raw_query
    try:
        if detect(raw_query) != "en":
            logging.info("Translating query to English...")
            trans_resp = _sync_openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": f"Translate to English: \"{raw_query}\""}]
            )
            query = trans_resp.choices[0].message.content.strip()
    except Exception:
        # If translation fails, just proceed
        pass

    # 2) Embed
    try:
        vec = _sync_openai.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        ).data[0].embedding
    except Exception as e:
        logging.error(f"OpenAI embedding failed: {e}")
        return []

    # 3) Search
    try:
        index = _pinecone_client.Index(index_name)
        resp = index.query(
            vector=vec,
            top_k=5,
            include_metadata=True,
            namespace=namespace
        )
        return resp.get("matches", [])
    except Exception as e:
        logging.error(f"Pinecone query failed: {e}")
        return []


def _rerank_results(
    results: List[Dict[str, Any]],
    structured_query: PolicyQuery
) -> List[Dict[str, Any]]:
    """
    Boosts result scores based on keywords from structured_query.
    """
    keywords = [
        structured_query.procedure_or_claim,
        structured_query.location
    ]
    keywords = [k.lower() for k in keywords if k]

    if not keywords:
        return results

    for r in results:
        text = r.get("metadata", {}).get("text", "").lower()
        bonus = sum(0.1 for kw in keywords if kw in text)
        r["score"] = r.get("score", 0) + bonus

    return sorted(results, key=lambda x: x["score"], reverse=True)
