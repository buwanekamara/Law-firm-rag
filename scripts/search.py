"""Debug CLI: retrieval only, no model involved.

    uv run scripts/search.py "confidentiality obligations"
    uv run scripts/search.py "termination notice" --k 10
    uv run scripts/search.py "payment terms" --doc hosting
    uv run scripts/search.py "force majeure" --text
    uv run scripts/search.py "the price per unit" --mode dense

Tells you whether a bad answer is retrieval's fault. If the right clause is
not in this list, no prompt will save you.

--mode overrides SEARCH_MODE for one search, which is the quickest way to see
what dense and sparse each contribute to a particular question.
"""

from __future__ import annotations

import argparse
import textwrap

from app.config import settings
from app.search.indexing import backend_description, collection_size, get_client
from app.search.retrieval import (
    SEARCH_MODES,
    infer_doc_filter,
    list_indexed_documents,
    search,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the contract index.")
    parser.add_argument("question", help="what to search for")
    parser.add_argument("--k", type=int, default=5, help="how many results (default 5)")
    parser.add_argument("--doc", help="restrict to one document id, or a fragment of one")
    parser.add_argument("--no-filter", action="store_true", help="disable automatic doc filtering")
    parser.add_argument("--text", action="store_true", help="print the matching text too")
    parser.add_argument(
        "--mode",
        choices=SEARCH_MODES,
        help=f"override SEARCH_MODE for this search (currently {settings.search_mode})",
    )
    args = parser.parse_args()

    client = get_client()
    if collection_size(client) == 0:
        print(f"The collection is empty ({backend_description()}).")
        print("Run: uv run scripts/index.py")
        return

    documents = list_indexed_documents(client)

    requested = None
    if args.doc:
        # Accept a fragment: --doc hosting means hosting_agreement.
        matches = [doc_id for doc_id in documents if args.doc.lower() in doc_id.lower()]
        if not matches:
            print(f"No indexed document matches {args.doc!r}. Known: {', '.join(sorted(documents))}")
            return
        if len(matches) > 1:
            print(f"{args.doc!r} matches several documents: {', '.join(sorted(matches))}")
            return
        requested = matches[0]

    inferred = None if args.no_filter else infer_doc_filter(args.question, documents)
    active = requested or inferred
    if active:
        source = "requested" if requested else "inferred from the question"
        print(f"Filtering to {active} ({source})\n")

    mode = args.mode or settings.search_mode
    results = search(
        args.question,
        top_k=args.k,
        doc_id=requested,
        auto_filter=not args.no_filter,
        client=client,
        mode=mode,
    )
    if not results:
        print("Nothing found.")
        return

    # Hybrid scores come from rank fusion, so they are not comparable with the
    # cosine similarities dense and sparse report.
    print(f"mode: {mode}\n")
    print(f"{'#':<3}{'score':<9}{'document':<30}{'section':<14}heading")
    print("-" * 100)
    for result in results:
        chunk = result.chunk
        print(
            f"{result.rank:<3}{result.score:<9.4f}{chunk['doc_title'][:28]:<30}"
            f"{chunk['section_label']:<14}{chunk['section_heading'][:34]}"
        )
        if args.text:
            body = textwrap.fill(chunk["text"], width=96, initial_indent="      ", subsequent_indent="      ")
            print(body[:1200] + ("..." if len(body) > 1200 else ""), "\n")


if __name__ == "__main__":
    main()
