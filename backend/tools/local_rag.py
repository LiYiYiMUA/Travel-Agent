"""
Utilities for local_expert RAG retrieval via Chroma Cloud.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import chromadb


logger = logging.getLogger("local_rag")


CITY_ALIASES = {
    "beijing": "beijing",
    "北京": "beijing",
    "shanghai": "shanghai",
    "上海": "shanghai",
    "guangzhou": "guangzhou",
    "广州": "guangzhou",
    "shenzhen": "shenzhen",
    "深圳": "shenzhen",
    "hangzhou": "hangzhou",
    "杭州": "hangzhou",
}


def normalize_city(city: str) -> str:
    key = (city or "").strip()
    if not key:
        return ""
    return CITY_ALIASES.get(key.lower(), CITY_ALIASES.get(key, key.lower()))


def get_collection_name() -> str:
    return os.getenv("CHROMA_COLLECTION", "travel_local_expert_knowledge").strip()


def get_default_top_k() -> int:
    value = os.getenv("CHROMA_TOP_K", "4").strip()
    try:
        top_k = int(value)
    except ValueError:
        top_k = 4
    return max(1, min(top_k, 10))


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.api.ClientAPI:
    api_key = _required_env("CHROMA_API_KEY")
    tenant = _required_env("CHROMA_TENANT")
    database = _required_env("CHROMA_DATABASE")
    return chromadb.CloudClient(api_key=api_key, tenant=tenant, database=database)


def _flatten_query_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs_nested = result.get("documents") or [[]]
    metas_nested = result.get("metadatas") or [[]]
    dists_nested = result.get("distances") or [[]]
    ids_nested = result.get("ids") or [[]]

    docs = docs_nested[0] if docs_nested else []
    metas = metas_nested[0] if metas_nested else []
    dists = dists_nested[0] if dists_nested else []
    ids = ids_nested[0] if ids_nested else []

    hits: List[Dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        meta = metas[idx] if idx < len(metas) and metas[idx] else {}
        distance = dists[idx] if idx < len(dists) else None
        chunk_id = ids[idx] if idx < len(ids) else None
        hits.append(
            {
                "id": chunk_id,
                "document": doc,
                "metadata": meta,
                "distance": distance,
            }
        )
    return hits


def query_local_knowledge(destination: str, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Query Chroma Cloud for local knowledge.
    Use city-filtered retrieval only to avoid cross-city contamination.
    """
    client = get_chroma_client()
    collection = client.get_collection(name=get_collection_name())
    n_results = top_k if top_k is not None else get_default_top_k()
    n_results = max(1, min(n_results, 10))

    city = normalize_city(destination)
    where = {"city": city} if city else None

    result = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = _flatten_query_result(result)
    return hits


def format_hits_for_llm(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return "未检索到本地知识库内容。"

    lines: List[str] = []
    for i, hit in enumerate(hits, 1):
        doc = (hit.get("document") or "").strip()
        meta = hit.get("metadata") or {}
        source = meta.get("source_file", "unknown")
        chunk_index = meta.get("chunk_index", "?")
        lines.append(
            f"{i}. [source={source}#chunk={chunk_index}]\n{doc}"
        )
    return "\n\n".join(lines)
