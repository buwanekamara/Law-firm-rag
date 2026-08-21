"""Debug CLI for phase 3 - retrieval only, no LLM involved.

    uv run scripts/search.py "confidentiality obligations"
    uv run scripts/search.py "termination notice" --k 10
    uv run scripts/search.py "payment terms" --doc hosting
    uv run scripts/search.py "force majeure" --text

This is the tool that tells you whether a bad answer is retrieval's fault.
If the right clause is not in this list, no prompt will save you.
"""

from __future__ import annotations

import argparse
import textwrap

from app.retrieval import infer_doc_filter, list_indexed_documents, search
from app.indexing import backend_description, collection_size, get_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the contract index.")
    parser.add_argument("question", help="what to search for")
    parser.add_argument("--k", type=int, default=5, help="how many results (default 5)")
    parser.add_argument("--doc", help="restrict to one document id")
    parser.add_argument("--no-filter", action="store_true", help="disable automatic doc filtering")
    parser.add_argument("--text", action="store_true", help="print the matching text too")
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

    results = search(
        args.question, top_k=args.k, doc_id=requested, auto_filter=not args.no_filter, client=client
    )
    if not results:
        print("Nothing found.")
        return

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
