"""Debug CLI for phase 1.

    uv run scripts/extract.py                 # all contracts
    uv run scripts/extract.py --doc hosting   # just one
    uv run scripts/extract.py --show 1        # print page 1 of each

Writes data/extracted/<doc_id>.json and a matching .txt you can read.
"""

from __future__ import annotations

import argparse

from app.extraction import extract_all
from app.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract contract PDFs to JSON.")
    parser.add_argument("--doc", help="only documents whose id contains this text")
    parser.add_argument("--show", type=int, metavar="PAGE", help="print this page after extracting")
    parser.add_argument("--no-txt", action="store_true", help="skip the .txt dumps")
    args = parser.parse_args()

    documents = extract_all(doc_filter=args.doc, write_txt=not args.no_txt)
    if not documents:
        print("No contracts matched. Is contracts/ populated?")
        return

    header = f"{'doc_id':<26}{'pages':>6}{'chars':>8}{'[***]':>7}{'[.]':>6}  title"
    print(header)
    print("-" * len(header))
    for doc in documents:
        s = doc["stats"]
        print(
            f"{doc['doc_id']:<26}{doc['page_count']:>6}{s['clean_chars']:>8}"
            f"{s['redaction_markers']:>7}{s['placeholder_markers']:>6}  {doc['title']}"
        )

    print("\nRemoved as page furniture (headers, footers, page numbers):")
    for doc in documents:
        s = doc["stats"]
        samples = ", ".join(repr(x) for x in s["dropped_samples"][:3]) or "nothing"
        print(f"  {doc['doc_id']:<26} {s['dropped_line_count']:>3} lines  e.g. {samples}")

    for doc in documents:
        s = doc["stats"]
        if s["redaction_markers"] != s["redaction_markers_raw"]:
            print(f"  !! {doc['doc_id']}: lost redaction markers during cleaning")
        if s["placeholder_markers"] != s["placeholder_markers_raw"]:
            print(f"  !! {doc['doc_id']}: lost placeholder markers during cleaning")

    if args.show:
        for doc in documents:
            page = next((p for p in doc["pages"] if p["page_no"] == args.show), None)
            print(f"\n{'=' * 90}\n{doc['title']} - page {args.show}\n{'=' * 90}")
            print(page["text"] if page else "(no such page)")

    print(f"\nWritten to {settings.extracted_dir}")


if __name__ == "__main__":
    main()
