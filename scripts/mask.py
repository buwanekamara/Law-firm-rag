"""Debug CLI for phase 6.5 - see exactly what masking would remove.

    uv run scripts/mask.py "Notices to smoore@penntex.com for Natalija Tunevic"
    uv run scripts/mask.py --corpus          # scan the indexed chunks
    uv run scripts/mask.py --terms           # show the protected defined terms

Masking is confusing to debug from inside an answer, which is why it has its
own switch (MASKING_ENABLED) and its own script.
"""

from __future__ import annotations

import argparse

from app.safety.masking import (
    MASKED_ENTITIES,
    MIN_CONFIDENCE,
    MaskingUnavailable,
    defined_terms,
    describe,
    mask_text,
    unmask_text,
)


def scan_corpus() -> None:
    from collections import Counter

    from app.ingest.chunking import load_chunks

    totals: Counter[str] = Counter()
    values: dict[str, set[str]] = {}
    for chunk in load_chunks():
        result = mask_text(chunk["text"])
        totals.update(result.entity_counts)
        for placeholder, original in result.mapping.items():
            values.setdefault(placeholder.rsplit("_", 1)[0], set()).add(original)

    if not totals:
        print("Nothing in the corpus would be masked.")
        return
    print("What masking would remove across all indexed chunks:\n")
    for entity, count in totals.most_common():
        found = sorted(values.get(entity, set()))
        print(f"  {entity:<16}{count:>4} hits   {', '.join(found[:6])}")
    print(
        "\nNot masked, by design: dates, locations, organisations, and any term the\n"
        "contracts define in quotation marks. See the docstring in app/masking.py."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PII masking.")
    parser.add_argument("text", nargs="?", help="text to mask")
    parser.add_argument("--corpus", action="store_true", help="scan every indexed chunk")
    parser.add_argument("--terms", action="store_true", help="list the protected defined terms")
    args = parser.parse_args()

    try:
        if args.terms:
            terms = sorted(defined_terms())
            print(f"{len(terms)} defined terms are protected from masking:\n")
            print(", ".join(terms))
            return

        if args.corpus:
            scan_corpus()
            return

        if not args.text:
            parser.error("give some text, or use --corpus / --terms")

        result = mask_text(args.text)
        print(f"\nentities considered: {', '.join(MASKED_ENTITIES)}")
        print(f"confidence floor:    {MIN_CONFIDENCE}\n")
        print("before:\n  " + args.text)
        print("\nafter:\n  " + result.text)
        print(f"\n{describe(result)}")
        for placeholder, original in sorted(result.mapping.items()):
            print(f"  {placeholder:<20} <- {original}")
        restored = unmask_text(result.text, result.mapping)
        print(f"\nround trip restores the original exactly: {restored == args.text}")
    except MaskingUnavailable as error:
        print(error)


if __name__ == "__main__":
    main()
