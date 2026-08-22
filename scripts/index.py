"""Build the Qdrant index from the chunk files.

    uv run scripts/index.py            # rebuild from scratch
    uv run scripts/index.py --keep     # add to the existing collection

Qdrant must be running first:

    docker run -p 6333:6333 -v ./qdrant_data:/qdrant/storage qdrant/qdrant

The first run downloads the embedding model (~130MB) and takes a minute or
two. After that it is seconds.
"""

from __future__ import annotations

import argparse
import time

from app.config import settings
from app.ingest.chunking import load_chunks
from app.search.indexing import backend_description, collection_size, get_client, index_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks and load them into Qdrant.")
    parser.add_argument("--keep", action="store_true", help="do not drop the existing collection")
    args = parser.parse_args()

    chunks = load_chunks()
    print(f"{len(chunks)} chunks to index -> {backend_description()}")
    print(f"collection: {settings.qdrant_collection}")
    print(f"dense: {settings.dense_model}   sparse: {settings.sparse_model}")

    client = get_client()
    started = time.perf_counter()
    written = index_chunks(chunks, client=client, recreate=not args.keep)
    elapsed = time.perf_counter() - started

    print(f"indexed {written} points in {elapsed:.1f}s")
    print(f"collection now holds {collection_size(client)} points")


if __name__ == "__main__":
    main()
