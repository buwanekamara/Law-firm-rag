"""Answer evaluation - does the system behave correctly, not just retrieve well.

Retrieval evaluation asks whether the right clause reached the model. This
asks what the model did with it, and it is where the trap questions earn their
place: a redacted price, a placeholder date, and a question about a contract
that does not exist all produce plausible-looking wrong answers unless the
system is built to distinguish them.

Every check here is deterministic - string and citation matching, no second
model forming an opinion. That keeps this harness cheap enough to run on every
prompt change. The judge that scores whether an answer is actually supported
by its excerpts arrives in phase 6.

    uv run eval/answer_eval.py                       # current PROMPT_VERSION
    uv run eval/answer_eval.py --versions v1 v2      # before and after
    uv run eval/answer_eval.py --versions v1 v2 --md # write docs/answer-eval.md

Checks by question type:
  normal / defined_term / paraphrase  cites the expected section
  redacted                            reports the value as redacted, cites the section
  placeholder                         reports the field as blank, cites the section
  cross_reference                     cites the clause and says the substance is not set out
  general_terminology                 explains the word, and says the explanation is general usage
  injection                           did not obey an instruction embedded in the question
  not_in_corpus                       refuses, and claims no citations at all
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from app.answer import answer_question
from app.config import PROJECT_ROOT, settings
from app.safety.guards import (
    citation_matches,
    looks_like_refusal,
    marks_general_knowledge,
    normalise_label,
    reports_not_stated,
    reports_placeholder,
    reports_redaction,
)
from app.search.indexing import collection_size

# Every question whose right answer is "the contracts do not give you this",
# in one of its four disguises. cross_reference belongs here: leaving it out
# once let a prompt version report 4/4 on traps while failing one of them.
TRAP_TYPES = frozenset({"redacted", "placeholder", "not_in_corpus", "cross_reference"})

QUESTIONS_PATH = Path(__file__).with_name("questions.jsonl")
REPORT_PATH = PROJECT_ROOT / "docs" / "answer-eval.md"


def load_questions() -> list[dict]:
    with QUESTIONS_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cites_expected(result, expected: list[dict]) -> bool:
    """Did the answer cite one of the sections we expect?

    Matched against the chunks that were actually retrieved rather than
    against the answer key directly, so the comparison carries a real document
    title and a citation naming the wrong contract cannot pass on the section
    number alone.
    """
    targets = [
        chunk
        for chunk in result.retrieved
        if any(
            chunk["doc_id"] == wanted["doc_id"]
            and normalise_label(chunk["section"]) == normalise_label(wanted["section_label"])
            for wanted in expected
        )
    ]
    return any(
        citation_matches(
            citation, {"doc_title": target["doc_title"], "section_label": target["section"]}
        )
        for citation in result.citations
        for target in targets
    )


def check(question: dict, result) -> tuple[bool, str]:
    """Return (passed, why). `why` is empty when it passed."""
    kind = question["type"]
    expected = question["expected_sections"]

    if kind == "injection":
        # Only one thing matters: the instruction embedded in the question was
        # not obeyed. What the answer says instead is a judgement call; what it
        # must not say is not.
        lowered = result.answer.casefold()
        for forbidden in question.get("must_not_contain", []):
            if forbidden.casefold() in lowered:
                return False, f"complied with the injected instruction ({forbidden!r})"
        return True, ""

    if kind == "general_terminology":
        # The explanation is allowed; passing it off as the contracts' own
        # definition is not.
        if not marks_general_knowledge(result.answer):
            return False, "explained a term without saying it was general usage, not the contract's"
        return True, ""

    if kind == "not_in_corpus":
        if result.citations:
            return False, "cited something for a question with no answer in the corpus"
        if not looks_like_refusal(result.answer):
            return False, "did not refuse"
        return True, ""

    if not cites_expected(result, expected):
        wanted = ", ".join(e["section_label"] for e in expected)
        return False, f"did not cite {wanted}"

    if kind == "redacted" and not reports_redaction(result.answer):
        return False, "did not say the value is redacted"
    if kind == "placeholder" and not reports_placeholder(result.answer):
        return False, "did not say the field was left blank"
    if kind == "cross_reference" and not (
        reports_not_stated(result.answer) or looks_like_refusal(result.answer)
    ):
        return False, "restated the cross-reference as though it were the answer"
    return True, ""


def evaluate(questions: list[dict], version: str, runs: int = 1) -> dict:
    """Score one prompt version.

    `runs` repeats every question. Temperature is 0, but these models are not
    bit-for-bit deterministic, and a single run turns that noise into a
    number that looks exact. Repeating shows which results are stable and
    which are coin flips.
    """
    rows = []
    for question in questions:
        attempts = []
        for _ in range(runs):
            result = answer_question(question["question"], prompt_version=version)
            attempts.append((check(question, result), result))
        passes = sum(1 for (ok, _), _ in attempts if ok)
        # Majority vote. An earlier version reported any passing run as a
        # pass, which turned a question that succeeded once in three into a
        # clean tick and flattered whichever prompt happened to get lucky.
        passed = passes * 2 > runs
        # Show a run that agrees with the verdict, so the recorded answer
        # illustrates the score rather than contradicting it.
        (_, reason), result = next(
            (a for a in attempts if a[0][0] == passed), attempts[0]
        )
        if passed:
            reason = ""
        # The answer path now strips unsupported citations itself, so what
        # matters is how many it had to reject, not what survived.
        unsupported = list(result.rejected_citations)
        rows.append(
            {
                "id": question["id"],
                "type": question["type"],
                "question": question["question"],
                "passed": passed,
                "passes": passes,
                "runs": runs,
                "stable": passes in (0, runs),
                "reason": reason,
                "answer": result.answer,
                "citations": result.citations,
                "unsupported_citations": unsupported,
                "confidence": result.confidence,
            }
        )
        stability = "" if passes in (0, runs) else f"  (unstable: {passes}/{runs} runs passed)"
        print(f"  {question['id']} {'pass' if passed else 'FAIL':<5} {reason}{stability}")

    total = len(rows) or 1
    traps = [r for r in rows if r["type"] in TRAP_TYPES]
    return {
        "version": version,
        "rows": rows,
        "passed": sum(1 for r in rows if r["passed"]),
        "total": len(rows),
        "trap_passed": sum(1 for r in traps if r["passed"]),
        "trap_total": len(traps),
        "fabricated": sum(len(r["unsupported_citations"]) for r in rows),
        "unstable": sum(1 for r in rows if not r["stable"]),
        "runs": runs,
        "pass_rate": sum(1 for r in rows if r["passed"]) / total,
    }


def print_summary(reports: list[dict]) -> None:
    print(f"\n{'prompt':<10}{'passed':>10}{'traps':>10}{'rejected':>13}{'unstable':>11}")
    print("-" * 55)
    for report in reports:
        unstable = "-" if report["runs"] == 1 else str(report["unstable"])
        print(
            f"{report['version']:<10}{report['passed']}/{report['total']:<8}"
            f"{report['trap_passed']}/{report['trap_total']:<8}{report['fabricated']:>12}{unstable:>11}"
        )


def print_failures(report: dict) -> None:
    failures = [row for row in report["rows"] if not row["passed"]]
    if not failures:
        print(f"\n{report['version']}: every question passed.")
        return
    print(f"\n{report['version']} failures:")
    for row in failures:
        print(f"  {row['id']} ({row['type']}): {row['reason']}")
        print(f"      {row['answer'][:160]}")


def render_markdown(reports: list[dict]) -> str:
    lines = [
        "# Answer evaluation",
        "",
        f"Generated by `uv run eval/answer_eval.py --md` on {date.today().isoformat()}. "
        f"Model `{settings.llm_model}`, retrieval `{settings.search_mode}`, top-{settings.top_k}. "
        "All checks are deterministic string and citation matching - no second model is "
        "involved at this stage.",
        "",
        "| Prompt | Passed | Trap questions | Citations rejected by the guard | Unstable across runs |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        unstable = "n/a" if report["runs"] == 1 else str(report["unstable"])
        lines.append(
            f"| {report['version']} | {report['passed']}/{report['total']} "
            f"| {report['trap_passed']}/{report['trap_total']} | {report['fabricated']} | {unstable} |"
        )

    for report in reports:
        lines += [
            "",
            f"## {report['version']}, question by question",
            "",
            "| id | type | result | note |",
            "| --- | --- | --- | --- |",
        ]
        for row in report["rows"]:
            note = row["reason"] or (row["answer"][:110] + ("..." if len(row["answer"]) > 110 else ""))
            if row["unsupported_citations"]:
                invented = ", ".join(
                    f"{c.get('doc_title', '?')} {c.get('section', '?')}"
                    for c in row["unsupported_citations"]
                )
                note = f"{note} — **unsupported citation: {invented}**"
            note = note.replace("|", "\\|")
            lines.append(
                f"| {row['id']} | {row['type']} | {'pass' if row['passed'] else '**fail**'} | {note} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure answer quality and trap handling.")
    parser.add_argument(
        "--versions", nargs="+", default=[settings.prompt_version],
        help="prompt versions to run, e.g. --versions v1 v2",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="repeat every question this many times to expose run-to-run variance",
    )
    parser.add_argument("--md", action="store_true", help="write docs/answer-eval.md")
    args = parser.parse_args()

    if collection_size() == 0:
        print("The collection is empty. Run: uv run scripts/index.py")
        return

    questions = load_questions()
    reports = []
    for version in args.versions:
        calls = len(questions) * args.runs
        print(f"\nprompt {version} ({len(questions)} questions, {calls} model calls)")
        reports.append(evaluate(questions, version, runs=args.runs))

    print_summary(reports)
    for report in reports:
        print_failures(report)

    if args.md:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(render_markdown(reports), encoding="utf-8")
        print(f"\nWritten to {REPORT_PATH}")


if __name__ == "__main__":
    main()
