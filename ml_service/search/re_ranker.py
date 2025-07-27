import logging
from typing import List, Dict, Any
import time
from functools import wraps
from query_parser.schema import PolicyQuery




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