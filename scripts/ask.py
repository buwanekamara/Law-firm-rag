"""Debug CLI for phase 4 - the whole pipeline, end to end.

    uv run scripts/ask.py "What are the confidentiality obligations?"
    uv run scripts/ask.py "When can the licence be terminated?" --debug
    uv run scripts/ask.py "What is the price?" --doc manufacturing --k 8
    uv run scripts/ask.py --models        # what the gateway will accept
    uv run scripts/ask.py --models openai # ...filtered

--debug prints the exact prompt that was sent and the raw reply. You will use
it constantly in the next two phases.
"""

from __future__ import annotations

import argparse
import textwrap

from app.answer import answer_question
from app.config import settings
from app.generate.llm import MissingApiKey, list_models
from app.search.indexing import collection_size, get_client
from app.search.retrieval import list_indexed_documents


def resolve_doc(fragment: str | None) -> str | None:
    if not fragment:
        return None
    documents = list_indexed_documents(get_client())
    matches = [doc_id for doc_id in documents if fragment.lower() in doc_id.lower()]
    if len(matches) != 1:
        known = ", ".join(sorted(documents))
        raise SystemExit(f"{fragment!r} matches {len(matches)} documents. Known: {known}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about the contracts.")
    parser.add_argument("question", nargs="?", help="the question")
    parser.add_argument("--k", type=int, help="how many excerpts to retrieve")
    parser.add_argument("--doc", help="restrict to one contract")
    parser.add_argument("--debug", action="store_true", help="show the prompt and raw reply")
    parser.add_argument(
        "--models", nargs="?", const="", metavar="FILTER",
        help="list gateway models and exit; optionally filter, e.g. --models openai",
    )
    args = parser.parse_args()

    if args.models is not None:
        try:
            models = list_models(args.models or None)
            for model in models:
                print(model)
            print(f"\n{len(models)} models")
        except MissingApiKey as error:
            print(error)
        return

    if not args.question:
        parser.error("a question is required (or use --models)")

    if collection_size() == 0:
        print("Nothing is indexed. Run: uv run scripts/index.py")
        return

    try:
        result = answer_question(
            args.question, top_k=args.k, doc_id=resolve_doc(args.doc), debug=args.debug
        )
    except MissingApiKey as error:
        print(error)
        return

    print(f"\nmodel: {settings.llm_model}   prompt: {settings.prompt_version}   "
          f"retrieval: {settings.search_mode}")
    print("=" * 96)
    print(textwrap.fill(result.answer, width=96))
    print("=" * 96)

    print(f"\nconfidence: {result.confidence}")
    if result.gate_score is not None:
        state = "GATED - no model call" if result.gated else "passed"
        print(f"relevance gate: {result.gate_score:.4f} ({state})")
    if result.parse_error:
        print(f"!! the reply could not be parsed as JSON: {result.parse_error}")
    for warning in result.warnings:
        print(f"!! {warning}")
    for rejected in result.rejected_citations:
        print(f"!! citation removed: {rejected.get('doc_title', '?')} {rejected.get('section', '?')}")

    print("\ncitations claimed by the model:")
    for citation in result.citations or []:
        page = f" p.{citation['page']}" if citation.get("page") else ""
        print(f"  - {citation.get('doc_title', '?')} | {citation.get('section', '?')}{page}")
    if not result.citations:
        print("  (none)")

    print("\nchunks actually retrieved:")
    for item in result.retrieved:
        print(
            f"  {item['rank']}. {item['score']:.4f}  {item['doc_title'][:30]:<32}"
            f"{item['section']:<14}p.{item['page_start']}"
        )

    if args.debug and result.debug:
        print("\n" + "=" * 96)
        print("SYSTEM PROMPT\n" + "=" * 96)
        print(result.debug["system_prompt"])
        print("\n" + "=" * 96)
        print("USER PROMPT\n" + "=" * 96)
        print(result.debug["user_prompt"])
        print("\n" + "=" * 96)
        print("RAW REPLY\n" + "=" * 96)
        print(result.debug["raw_response"])


if __name__ == "__main__":
    main()
