# ml_service/search.py

import os
import logging
from typing import List, Dict, Any, Optional, Awaitable

import pinecone
from openai import OpenAI
from langdetect import detect
from query_parser.schema import PolicyQuery

# — Sync clients (singleton pattern) —
import sys

def _get_env_var(key: str) -> str:
    value = os.environ.get(key)
    if value is None:
        logging.error(f"Environment variable '{key}' not set.")
        sys.exit(1)
    return value

try:
    _sync_openai: Optional[OpenAI] = OpenAI(api_key=_get_env_var("OPENAI_API_KEY"))
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")
    _sync_openai = None

try:
    _pinecone_client: Optional[pinecone.Pinecone] = pinecone.Pinecone(api_key=_get_env_var("PINECONE_API_KEY"))
except Exception as e:
    logging.error(f"Failed to initialize Pinecone client: {e}")
    _pinecone_client = None

# — Async clients: you should import and pass these from your `clients.py` —
# from clients import pinecone_client, openai_async_client


def _sync_embed_query(query: str) -> Any:
    """
    Synchronously embed a query string using the OpenAI client.
    Returns the embedding vector or an error dict.
    """
    try:
        return _sync_openai.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        ).data[0].embedding
    except Exception as e:
        logging.error(f"OpenAI embedding failed: {e}")
        return {"error": f"OpenAI embedding failed: {e}"}

def _sync_pinecone_query(vec: Any, index_name: str, namespace: Optional[str], top_k: int) -> Any:
    """
    Synchronously query Pinecone with the given vector.
    Returns a list of matches or an error dict.
    """
    try:
        index = _pinecone_client.Index(index_name)
        resp = index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        return resp.get("matches", [])
    except Exception as e:
        logging.error(f"Pinecone query failed: {e}")
        return [{"error": f"Pinecone query failed: {e}"}]

async def _async_embed_query(query: str, openai_client: OpenAI) -> Any:
    """
    Asynchronously embed a query string using the provided OpenAI client.
    Returns the embedding vector or an error dict.
    """
    try:
        resp = await openai_client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        )
        return resp.data[0].embedding
    except Exception as e:
        logging.error(f"[Async] OpenAI embedding failed: {e}")
        return {"error": f"OpenAI embedding failed: {e}"}

def _async_pinecone_query(vec: Any, pinecone_client: pinecone.Pinecone, index_name: str, namespace: str, top_k: int) -> Any:
    """
    Asynchronously query Pinecone with the given vector.
    Returns a list of matches or an error dict.
    """
    try:
        index = pinecone_client.Index(index_name)
        response = index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        return response.get("matches", [])
    except Exception as e:
        logging.error(f"[Async] Pinecone query failed: {e}")
        return [{"error": f"Pinecone query failed: {e}"}]

async def perform_search_async(
    raw_query: str,
    index_name: str,
    namespace: str,
    pinecone_client: pinecone.Pinecone,
    openai_client: OpenAI,
    top_k: int = 3,
    structured_query: Optional[PolicyQuery] = None
) -> List[Dict[str, Any]]:
    """
    Asynchronously embeds the query and runs a Pinecone search.
    Optionally reranks results using structured_query.
    Returns a list of matches or a dict with an 'error' key if something fails.
    Args:
        raw_query: The query string to search for.
        index_name: The Pinecone index name.
        namespace: The Pinecone namespace.
        pinecone_client: The Pinecone client instance.
        openai_client: The OpenAI client instance.
        top_k: Number of results to return (default 3).
        structured_query: Optional PolicyQuery for reranking.
    Returns:
        List of result dicts or error dicts.
    """
    vec = await _async_embed_query(raw_query, openai_client)
    if isinstance(vec, dict) and "error" in vec:
        return [vec]
    results = _async_pinecone_query(vec, pinecone_client, index_name, namespace, top_k)
    if structured_query and isinstance(results, list) and results and "error" not in results[0]:
        results = _rerank_results(results, structured_query)
    return results

def perform_search(
    raw_query: str,
    index_name: str,
    namespace: Optional[str] = None,
    top_k: int = 5,
    structured_query: Optional[PolicyQuery] = None
) -> List[Dict[str, Any]]:
    """
    Synchronously embeds the query and runs a Pinecone search.
    Optionally reranks results using structured_query.
    Returns a list of matches or a dict with an 'error' key if something fails.
    Args:
        raw_query: The query string to search for.
        index_name: The Pinecone index name.
        namespace: The Pinecone namespace (optional).
        top_k: Number of results to return (default 5).
        structured_query: Optional PolicyQuery for reranking.
    Returns:
        List of result dicts or error dicts.
    """
    if not _sync_openai or not _pinecone_client:
        logging.error("Sync clients not initialized.")
        return [{"error": "Sync clients not initialized."}]

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
    except Exception as e:
        logging.error(f"Translation failed: {e}")
        return [{"error": f"Translation failed: {e}"}]

    vec = _sync_embed_query(query)
    if isinstance(vec, dict) and "error" in vec:
        return [vec]
    results = _sync_pinecone_query(vec, index_name, namespace, top_k)
    if structured_query and isinstance(results, list) and results and "error" not in results[0]:
        results = _rerank_results(results, structured_query)
    return results


def _rerank_results(
    results: List[Dict[str, Any]],
    structured_query: PolicyQuery
) -> List[Dict[str, Any]]:
    """
    Boosts result scores based on keywords from structured_query.
    Args:
        results: List of Pinecone result dicts.
        structured_query: PolicyQuery object with fields to boost.
    Returns:
        List of result dicts, reranked by score.
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