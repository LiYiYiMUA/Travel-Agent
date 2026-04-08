#!/usr/bin/env python3
"""
Chunk markdown files in knowledge-rag and upload them to Chroma Cloud.

Usage example:
  python backend/scripts/ingest_local_knowledge_to_chroma.py \
    --knowledge-dir ../knowledge-rag \
    --api-key <CHROMA_API_KEY> \
    --tenant <CHROMA_TENANT> \
    --database multi-agent \
    --collection travel_local_expert_knowledge
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb


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


def split_markdown_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    clean = text.replace("\r\n", "\n").strip()
    if not clean:
        return []

    chunks: List[str] = []
    start = 0
    total = len(clean)
    min_break_pos = int(chunk_size * 0.6)

    while start < total:
        end = min(start + chunk_size, total)
        if end < total:
            window = clean[start:end]
            break_pos = max(
                window.rfind("\n\n"),
                window.rfind("\n"),
                window.rfind("。"),
                window.rfind("！"),
                window.rfind("？"),
                window.rfind(". "),
            )
            if break_pos >= min_break_pos:
                end = start + break_pos + 1

        if end <= start:
            end = min(start + chunk_size, total)

        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= total:
            break

        start = max(start + 1, end - chunk_overlap)

    return chunks


def build_documents(
    knowledge_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[str]]:
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    source_files: List[str] = []
    ingest_time = datetime.now(timezone.utc).isoformat()

    md_files = sorted(knowledge_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown files found in: {knowledge_dir}")

    for md_file in md_files:
        source_file = md_file.name
        city = normalize_city(md_file.stem)
        source_files.append(source_file)
        text = md_file.read_text(encoding="utf-8")
        chunks = split_markdown_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for idx, chunk in enumerate(chunks):
            raw_id = f"{source_file}:{idx}:{chunk}"
            digest = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
            doc_id = f"{city}_{idx}_{digest}"

            ids.append(doc_id)
            docs.append(chunk)
            metas.append(
                {
                    "city": city,
                    "source_file": source_file,
                    "chunk_index": idx,
                    "chunk_length": len(chunk),
                    "ingested_at": ingest_time,
                }
            )

    return ids, docs, metas, sorted(set(source_files))


def get_required(value: str, env_name: str) -> str:
    if value:
        return value
    env_value = os.getenv(env_name, "").strip()
    if not env_value:
        raise ValueError(f"Missing required parameter: --{env_name.lower()} or {env_name}")
    return env_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local markdown knowledge into Chroma Cloud.")
    parser.add_argument(
        "--knowledge-dir",
        default=str(Path(__file__).resolve().parents[2] / "knowledge-rag"),
        help="Directory containing city markdown files.",
    )
    parser.add_argument("--api-key", default="", help="Chroma Cloud API key.")
    parser.add_argument("--tenant", default="", help="Chroma Cloud tenant id.")
    parser.add_argument("--database", default="", help="Chroma Cloud database name.")
    parser.add_argument(
        "--collection",
        default=os.getenv("CHROMA_COLLECTION", "travel_local_expert_knowledge"),
        help="Target Chroma collection.",
    )
    parser.add_argument("--chunk-size", type=int, default=900, help="Chunk size in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap in characters.")
    parser.add_argument("--batch-size", type=int, default=64, help="Upload batch size.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the target collection before ingesting.",
    )
    args = parser.parse_args()

    api_key = get_required(args.api_key.strip(), "CHROMA_API_KEY")
    tenant = get_required(args.tenant.strip(), "CHROMA_TENANT")
    database = get_required(args.database.strip(), "CHROMA_DATABASE")

    knowledge_dir = Path(args.knowledge_dir).resolve()
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"knowledge directory does not exist: {knowledge_dir}")

    ids, docs, metas, source_files = build_documents(
        knowledge_dir=knowledge_dir,
        chunk_size=max(200, args.chunk_size),
        chunk_overlap=max(0, args.chunk_overlap),
    )
    print(f"Prepared {len(docs)} chunks from {len(source_files)} markdown files.")

    client = chromadb.CloudClient(api_key=api_key, tenant=tenant, database=database)
    collection_name = args.collection.strip()

    if args.recreate:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass

    collection = client.get_or_create_collection(name=collection_name)

    # Remove old chunks for these source files to avoid duplicates.
    for source_file in source_files:
        try:
            collection.delete(where={"source_file": source_file})
        except Exception:
            # Ignore "no existing records" style failures.
            pass

    batch_size = max(1, args.batch_size)
    for i in range(0, len(ids), batch_size):
        end = i + batch_size
        collection.add(
            ids=ids[i:end],
            documents=docs[i:end],
            metadatas=metas[i:end],
        )
        print(f"Uploaded chunks: {i} -> {min(end, len(ids))}")

    total_count = collection.count()
    print(f"Done. Collection '{collection_name}' now has {total_count} records.")


if __name__ == "__main__":
    main()
