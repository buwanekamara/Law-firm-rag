"""Retrieval evaluation - run before any LLM is involved.

Measures whether the right clause reaches the model at all. Separating this
from answer quality is the whole point: when an answer is wrong, this number
says immediately whether retrieval or generation is to blame.

    uv run eval/retrieval_eval.py
    uv run eval/retrieval_eval.py --k 10
    uv run eval/retrieval_eval.py --md      # write docs/retrieval-eval.md

Metrics:
  hit@k  the share of questions where a correct section appears in the top k.
  MRR    mean reciprocal rank - 1/rank of the first correct hit, averaged.
         Rewards putting the right clause first, not merely somewhere.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from app.config import PROJECT_ROOT, settings
from app.search.indexing import collection_size, get_client
from app.search.retrieval import SEARCH_MODES, infer_doc_filter, list_indexed_documents, search

QUESTIONS_PATH = Path(__file__).with_name("questions.jsonl")
REPORT_PATH = PROJECT_ROOT / "docs" / "retrieval-eval.md"


def load_questions() -> list[dict]:
    with QUESTIONS_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def first_correct_rank(results, expected: list[dict]) -> int | None:
    """Rank (1-based) of the first result matching any expected section."""
    wanted = {(e["doc_id"], e["section_label"]) for e in expected}
    for result in results:
        key = (result.chunk["doc_id"], result.chunk["section_label"])
        if key in wanted:
            return result.rank
    return None


def evaluate(questions: list[dict], mode: str, top_k: int, client, auto_filter: bool = True) -> dict:
    """Score one retrieval mode.

    Questions with no expected sections - those asking about something that is
    not in the corpus at all - are reported but excluded from hit@k. Nothing
    can be retrieved correctly for them, so counting them as misses would
    penalise retrieval for a case that belongs to the refusal guards instead.
    """
    rows = []
    for question in questions:
        results = search(
            question["question"], top_k=top_k, client=client, mode=mode, auto_filter=auto_filter
        )
        scored = bool(question["expected_sections"])
        rank = first_correct_rank(results, question["expected_sections"]) if scored else None
        rows.append(
            {
                "id": question["id"],
                "type": question["type"],
                "question": question["question"],
                "scored": scored,
                "rank": rank,
                "top_hit": (
                    f"{results[0].chunk['doc_title']} {results[0].chunk['section_label']}"
                    if results
                    else "-"
                ),
            }
        )

    scored_rows = [row for row in rows if row["scored"]]
    total = len(scored_rows) or 1
    return {
        "mode": mode,
        "auto_filter": auto_filter,
        "rows": rows,
        "scored": len(scored_rows),
        "hit@1": sum(1 for r in scored_rows if r["rank"] == 1) / total,
        "hit@3": sum(1 for r in scored_rows if r["rank"] and r["rank"] <= 3) / total,
        "hit@5": sum(1 for r in scored_rows if r["rank"] and r["rank"] <= 5) / total,
        "mrr": sum(1 / r["rank"] for r in scored_rows if r["rank"]) / total,
    }


def print_summary(reports: list[dict], total: int, title: str) -> None:
    print(f"\n{title}")
    print(f"{'mode':<10}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR':>8}")
    print("-" * 42)
    for report in reports:
        print(
            f"{report['mode']:<10}{report['hit@1']:>7.0%}{report['hit@3']:>8.0%}"
            f"{report['hit@5']:>8.0%}{report['mrr']:>8.2f}"
        )
    print(f"({total} questions)")


def print_detail(report: dict) -> None:
    setting = "with filter" if report["auto_filter"] else "no filter"
    print(f"\nPer question ({report['mode']}, {setting}):")
    print(f"{'id':<5}{'rank':<6}{'type':<14}question")
    print("-" * 100)
    for row in report["rows"]:
        rank = "n/a" if not row["scored"] else (str(row["rank"]) if row["rank"] else "MISS")
        print(f"{row['id']:<5}{rank:<6}{row['type']:<14}{row['question'][:64]}")


def print_comparison(reports: list[dict]) -> None:
    """Per-question ranks for every mode, side by side.

    Aggregate scores can tie while the modes fail on completely different
    questions - which is the case for hybrid retrieval. This table is what
    shows it.
    """
    by_mode = {report["mode"]: {row["id"]: row for row in report["rows"]} for report in reports}
    if len(by_mode) < 2:
        return
    modes = list(by_mode)
    ids = [row["id"] for row in reports[0]["rows"]]

    print("\nRank of the first correct section, by mode (no filter):")
    header = f"{'id':<5}{'type':<14}" + "".join(f"{mode:>9}" for mode in modes) + "   question"
    print(header)
    print("-" * 104)
    for question_id in ids:
        row = by_mode[modes[0]][question_id]
        def cell(mode: str, question_id: str = question_id) -> str:
            entry = by_mode[mode][question_id]
            return "n/a" if not entry["scored"] else str(entry["rank"] or "miss")

        ranks = "".join(f"{cell(mode):>9}" for mode in modes)
        marker = ""
        values = [by_mode[mode][question_id]["rank"] for mode in modes]
        if row["scored"] and any(v is None or v > 3 for v in values) and any(v == 1 for v in values):
            marker = "  <- modes disagree"
        print(f"{row['id']:<5}{row['type']:<14}{ranks}   {row['question'][:46]}{marker}")


def render_markdown(with_filter: list[dict], without_filter: list[dict], top_k: int) -> str:
    detail = next(r for r in with_filter if r["mode"] == "hybrid")
    lines = [
        "# Retrieval evaluation",
        "",
        f"Generated by `uv run eval/retrieval_eval.py --md` on {date.today().isoformat()}. "
        f"{len(detail['rows'])} questions ({detail['scored']} scored for retrieval), top-{top_k}. "
        "No LLM is involved in these numbers.",
        "",
        "Two settings are reported. **With the document filter** is how the system actually "
        "behaves: a question naming one contract is restricted to it, which shrinks the "
        "haystack and flatters every retrieval mode. **Without the filter** every question "
        "searches all 120 chunks, which is the honest comparison between dense, sparse and "
        "hybrid.",
        "",
        "## With the document filter (production behaviour)",
        "",
        "| Mode | hit@1 | hit@3 | hit@5 | MRR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for report in with_filter:
        lines.append(
            f"| {report['mode']} | {report['hit@1']:.0%} | {report['hit@3']:.0%} "
            f"| {report['hit@5']:.0%} | {report['mrr']:.2f} |"
        )
    lines += [
        "",
        "## Without the document filter (all 120 chunks searched)",
        "",
        "| Mode | hit@1 | hit@3 | hit@5 | MRR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for report in without_filter:
        lines.append(
            f"| {report['mode']} | {report['hit@1']:.0%} | {report['hit@3']:.0%} "
            f"| {report['hit@5']:.0%} | {report['mrr']:.2f} |"
        )
    modes_present = [report["mode"] for report in without_filter]
    if len(modes_present) > 1:
        by_mode = {
            report["mode"]: {row["id"]: row for row in report["rows"]}
            for report in without_filter
        }
        lines += [
            "",
            "## Where the modes disagree (no filter)",
            "",
            "Rank of the first correct section. Aggregate scores can tie while the two halves "
            "fail on different questions - that is what this table is for.",
            "",
            "| id | type | " + " | ".join(modes_present) + " | question |",
            "| --- | --- | " + " | ".join("---:" for _ in modes_present) + " | --- |",
        ]
        for row in without_filter[0]["rows"]:
            ranks = " | ".join(
                "n/a" if not by_mode[mode][row["id"]]["scored"]
                else str(by_mode[mode][row["id"]]["rank"] or "miss")
                for mode in modes_present
            )
            lines.append(f"| {row['id']} | {row['type']} | {ranks} | {row['question']} |")

    lines += [
        "",
        "## Per question (hybrid)",
        "",
        "| id | type | rank of first correct section | question |",
        "| --- | --- | ---: | --- |",
    ]
    for row in detail["rows"]:
        rank = "n/a" if not row["scored"] else (row["rank"] or "miss")
        lines.append(f"| {row['id']} | {row['type']} | {rank} | {row['question']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure retrieval quality.")
    parser.add_argument(
        "--k", type=int, default=settings.top_k,
        help=f"results per question (default {settings.top_k}, matching TOP_K)",
    )
    parser.add_argument("--mode", choices=[*SEARCH_MODES, "all"], default="all")
    parser.add_argument("--md", action="store_true", help="write docs/retrieval-eval.md")
    args = parser.parse_args()

    client = get_client()
    if collection_size(client) == 0:
        print("The collection is empty. Run: uv run scripts/index.py")
        return

    questions = load_questions()
    modes = SEARCH_MODES if args.mode == "all" else (args.mode,)

    with_filter = [evaluate(questions, mode, args.k, client, auto_filter=True) for mode in modes]
    without_filter = [evaluate(questions, mode, args.k, client, auto_filter=False) for mode in modes]

    documents = list_indexed_documents(client)
    touched = sum(1 for q in questions if infer_doc_filter(q["question"], documents))

    print_summary(
        with_filter,
        len(questions),
        f"With the document filter (production behaviour; {touched} of {len(questions)} "
        "questions name a contract and are narrowed to it)",
    )
    print_summary(
        without_filter, len(questions), "Without the document filter (all chunks searched)"
    )

    print_comparison(without_filter)

    shown = "hybrid" if "hybrid" in modes else modes[0]
    print_detail(next(r for r in without_filter if r["mode"] == shown))

    if args.md:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(render_markdown(with_filter, without_filter, args.k), encoding="utf-8")
        print(f"\nWritten to {REPORT_PATH}")


if __name__ == "__main__":
    main()
