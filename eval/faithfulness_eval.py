"""LLM-as-judge faithfulness scoring.

The checks in answer_eval.py are deterministic: did the answer cite the right
section, did it use the right words for a redaction. They cannot tell whether
the *substance* of an answer is actually supported by the excerpts it was
given. That needs something that can read.

Method, a minimal reimplementation of the faithfulness metric used by RAGAS:
break each answer into atomic claims, ask whether each claim is supported by
the retrieved excerpts, and score the answer as supported claims over total
claims. The score says nothing about whether the answer is *correct* in the
world - only whether it stayed inside its sources, which is exactly what
hallucination means here.

The judge is a different model from the answerer (JUDGE_MODEL, not
LLM_MODEL), because a model grading its own output tends to agree with itself.
That removes the most obvious form of the bias, not all of it: two models can
share a blind spot, so this is a check rather than a proof.

    uv run eval/faithfulness_eval.py
    uv run eval/faithfulness_eval.py --md
    uv run eval/faithfulness_eval.py --ids t01 t04
    uv run eval/faithfulness_eval.py --selftest   # can the judge fail anything?

The self-test exists because a perfect score proves nothing on its own: a
judge that always answers "supported" gives exactly the output a good judge
gives on clean answers. --selftest shows it answers with known invented claims
that it has to catch, plus one clean answer it has to leave alone.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from app.answer import answer_question, extract_json
from app.config import PROJECT_ROOT, settings
from app.generate.llm import MissingApiKey, complete
from app.generate.prompting import render
from app.search.indexing import collection_size

QUESTIONS_PATH = Path(__file__).with_name("questions.jsonl")
REPORT_PATH = PROJECT_ROOT / "docs" / "faithfulness-eval.md"

JUDGE_PROMPT = "judge_v1"


# Each control pairs a real question - so the excerpts are genuine - with an
# answer containing one specific fabrication. The judge has to find it.
NEGATIVE_CONTROLS = (
    {
        "name": "invented figure",
        "question": "How much does the client pay for the website?",
        "answer": (
            "The Client agrees to pay the Company a total of $5,000 for the design and "
            "development of the website (Section 1). A late payment fee of 4% per month "
            "applies to any overdue balance."
        ),
        "planted": "a late payment fee of 4% per month, which appears nowhere in the hosting agreement",
    },
    {
        "name": "right clause, wrong contract",
        "question": "Who bears the risk of loss for products during shipment?",
        "answer": (
            "Under the Hosting Agreement, Heritage bears the risk of loss to the Products "
            "until they are delivered to a carrier for delivery to Premier (Section 8)."
        ),
        "planted": "attributing a manufacturing agreement clause to the hosting agreement",
    },
    {
        "name": "inference from silence",
        "question": "Who owns the written material the client supplies for the website?",
        "answer": (
            "The Client retains full ownership of all written material supplied for the "
            "website, and the Company acquires no rights in it whatsoever, because the "
            "agreement does not provide for any transfer."
        ),
        "planted": "a conclusion about ownership drawn from the contract's silence",
    },
    {
        "name": "plausible but contradicted",
        "question": "Can the licensor terminate the trademark licence for convenience?",
        "answer": (
            "Either party may terminate the Trademark License Agreement for convenience on "
            "thirty days' written notice (Section 4.2)."
        ),
        "planted": "a thirty-day notice period, where the clause says termination is immediate",
    },
)

# And one answer that is entirely supported. A judge that flags this too is
# not strict - it is broken in the other direction.
POSITIVE_CONTROL = {
    "name": "clean answer",
    "question": "Which state's law governs the manufacturing agreement?",
    "answer": (
        "The Manufacturing Agreement is governed by the laws of the State of California "
        "(Section 12)."
    ),
}


def excerpts_for(question: str) -> str:
    """The real retrieved excerpts for a question, as the answerer would see them."""
    from app.answer import build_excerpts
    from app.search.retrieval import search

    return build_excerpts(search(question, top_k=settings.top_k))


def run_selftest() -> bool:
    """Show the judge known-bad answers. Returns True if it caught them all."""
    print(f"Judge: {settings.judge_model}\n")
    print(f"{'control':<28}{'claims':>8}{'flagged':>9}   verdict")
    print("-" * 92)

    caught = 0
    for control in NEGATIVE_CONTROLS:
        claims = judge(control["answer"], excerpts_for(control["question"]))
        unsupported = [c for c in claims if not c.get("supported")]
        found = bool(unsupported)
        caught += found
        verdict = "caught" if found else "MISSED"
        print(f"{control['name']:<28}{len(claims):>8}{len(unsupported):>9}   {verdict}")
        if found:
            print(f"{'':<28}{'':>17}   -> {unsupported[0]['claim'][:60]}")
        else:
            print(f"{'':<28}{'':>17}   -> planted: {control['planted']}")

    claims = judge(POSITIVE_CONTROL["answer"], excerpts_for(POSITIVE_CONTROL["question"]))
    unsupported = [c for c in claims if not c.get("supported")]
    clean_ok = not unsupported
    print(
        f"{POSITIVE_CONTROL['name']:<28}{len(claims):>8}{len(unsupported):>9}   "
        + ("left alone" if clean_ok else "FALSE POSITIVE")
    )
    if not clean_ok:
        print(f"{'':<28}{'':>17}   -> {unsupported[0]['claim'][:60]}")

    print()
    if caught == len(NEGATIVE_CONTROLS) and clean_ok:
        print(
            f"The judge caught {caught}/{len(NEGATIVE_CONTROLS)} planted fabrications and left the\n"
            "clean answer alone. Its scores mean something."
        )
        return True

    print(
        f"The judge caught {caught}/{len(NEGATIVE_CONTROLS)} planted fabrications"
        + ("" if clean_ok else " and flagged a clean answer")
        + ".\nTreat its scores with caution and say so in any report that quotes them."
    )
    return False


def load_questions(ids: list[str] | None = None) -> list[dict]:
    with QUESTIONS_PATH.open(encoding="utf-8") as handle:
        questions = [json.loads(line) for line in handle if line.strip()]
    if ids:
        questions = [q for q in questions if q["id"] in set(ids)]
    return questions


def judge(answer: str, excerpts: str) -> list[dict]:
    """Ask the judge model to decompose and check an answer."""
    system, user = render(JUDGE_PROMPT, EXCERPTS=excerpts, ANSWER=answer)
    raw = complete(system, user, model=settings.judge_model)
    payload = extract_json(raw)
    claims = payload.get("claims", [])
    return [claim for claim in claims if isinstance(claim, dict) and "claim" in claim]


def score_question(question: dict, runs: int = 1) -> dict:
    """Score one question, averaged over `runs` independent attempts.

    Two things vary and they compound: the answering model is not deterministic
    even at temperature 0, and the judge is a second model reading whatever
    came out. The same question can yield 5 claims at 60% on one run and 3 at
    100% on the next, so a single sample is not a measurement.
    """
    attempts = [_score_once(question) for _ in range(runs)]
    scored = [attempt for attempt in attempts if attempt["faithfulness"] is not None]
    if not scored:
        return {**attempts[0], "runs": runs, "spread": None}

    scores = [attempt["faithfulness"] for attempt in scored]
    best = min(scored, key=lambda attempt: attempt["faithfulness"])  # show the worst run
    return {
        **best,
        "runs": runs,
        "faithfulness": sum(scores) / len(scores),
        "spread": (min(scores), max(scores)) if len(scores) > 1 else None,
    }


def _score_once(question: dict) -> dict:
    result = answer_question(question["question"], debug=True)
    excerpts = "\n\n".join(
        f"{chunk['header']}\n{chunk['text']}" for chunk in (result.debug or {}).get("chunks", [])
    )

    if not excerpts or result.gated:
        return {
            "id": question["id"],
            "type": question["type"],
            "question": question["question"],
            "answer": result.answer,
            "claims": [],
            "supported": 0,
            "total": 0,
            "faithfulness": None,
            "note": "gated or nothing retrieved - no claims to check",
        }

    try:
        claims = judge(result.answer, excerpts)
    except (ValueError, json.JSONDecodeError) as error:
        return {
            "id": question["id"],
            "type": question["type"],
            "question": question["question"],
            "answer": result.answer,
            "claims": [],
            "supported": 0,
            "total": 0,
            "faithfulness": None,
            "note": f"judge reply could not be parsed: {error}",
        }

    supported = sum(1 for claim in claims if claim.get("supported"))
    return {
        "id": question["id"],
        "type": question["type"],
        "question": question["question"],
        "answer": result.answer,
        "claims": claims,
        "supported": supported,
        "total": len(claims),
        "faithfulness": (supported / len(claims)) if claims else None,
        "note": "",
    }


def print_rows(rows: list[dict]) -> None:
    multi = any(row.get("runs", 1) > 1 for row in rows)
    header = f"\n{'id':<6}{'claims':>8}{'supported':>11}{'faithful':>10}"
    header += f"{'range':>14}" if multi else ""
    print(header + "  unsupported claims")
    print("-" * 110)
    for row in rows:
        score = "-" if row["faithfulness"] is None else f"{row['faithfulness']:.0%}"
        bad = [c["claim"] for c in row["claims"] if not c.get("supported")]
        summary = bad[0][:44] + ("..." if len(bad[0]) > 44 else "") if bad else row["note"]
        spread = ""
        if multi:
            pair = row.get("spread")
            spread = f"{pair[0]:.0%}-{pair[1]:.0%}" if pair else ""
            spread = f"{spread:>14}"
        print(f"{row['id']:<6}{row['total']:>8}{row['supported']:>11}{score:>10}{spread}  {summary}")


def summarise(rows: list[dict]) -> dict:
    scored = [row for row in rows if row["faithfulness"] is not None]
    total_claims = sum(row["total"] for row in scored)
    supported = sum(row["supported"] for row in scored)
    return {
        "questions": len(rows),
        "scored": len(scored),
        "claims": total_claims,
        "supported": supported,
        "mean": (sum(row["faithfulness"] for row in scored) / len(scored)) if scored else 0.0,
        "perfect": sum(1 for row in scored if row["faithfulness"] == 1.0),
        "runs": max((row.get("runs", 1) for row in rows), default=1),
        "unstable": sum(1 for row in scored if row.get("spread") and row["spread"][0] != row["spread"][1]),
    }


def render_markdown(rows: list[dict], summary: dict) -> str:
    lines = [
        "# Faithfulness evaluation",
        "",
        f"Generated by `uv run eval/faithfulness_eval.py --md` on {date.today().isoformat()}.",
        "",
        f"Answers from `{settings.llm_model}` (prompt **{settings.prompt_version}**), judged by "
        f"`{settings.judge_model}`. A different model does the judging so the answering model "
        "is not grading its own work.",
        "",
        "Each answer is broken into atomic claims; each claim is checked against the excerpts "
        "that answer was built from. Faithfulness is supported claims over total claims. It "
        "measures whether an answer stayed inside its sources - not whether it is correct in "
        "the world.",
        "",
        f"- Questions scored: **{summary['scored']}/{summary['questions']}**",
        f"- Claims checked: **{summary['claims']}**, supported: **{summary['supported']}**",
        f"- Mean faithfulness: **{summary['mean']:.1%}**",
        f"- Answers with every claim supported: **{summary['perfect']}/{summary['scored']}**",
        (
            f"- Runs per question: **{summary['runs']}**"
            + (f", of which **{summary['unstable']}** scored differently between runs"
               if summary["runs"] > 1 else "")
        ),
        "",
        "| id | type | claims | supported | faithfulness | unsupported claim |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        score = "-" if row["faithfulness"] is None else f"{row['faithfulness']:.0%}"
        bad = [c["claim"] for c in row["claims"] if not c.get("supported")]
        note = (bad[0] if bad else row["note"]).replace("|", "\\|")
        lines.append(
            f"| {row['id']} | {row['type']} | {row['total']} | {row['supported']} "
            f"| {score} | {note} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score answers for faithfulness to their sources.")
    parser.add_argument("--ids", nargs="+", help="only these question ids")
    parser.add_argument(
        "--runs", type=int, default=1,
        help="score every question this many times and average - the answer and the "
        "judge both vary between runs",
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="check the judge can detect fabrications, before trusting its scores",
    )
    parser.add_argument("--md", action="store_true", help="write docs/faithfulness-eval.md")
    args = parser.parse_args()

    if collection_size() == 0:
        print("The collection is empty. Run: uv run scripts/index.py")
        return

    if args.selftest:
        try:
            run_selftest()
        except MissingApiKey as error:
            print(error)
        return

    if settings.judge_model == settings.llm_model:
        print(
            f"!! JUDGE_MODEL and LLM_MODEL are both {settings.llm_model}.\n"
            "   A model grading its own answers inflates the score. Set a different "
            "JUDGE_MODEL in .env.\n"
        )

    questions = load_questions(args.ids)
    print(f"answering with {settings.llm_model} (prompt {settings.prompt_version}), "
          f"judging with {settings.judge_model}")
    calls = len(questions) * 2 * args.runs
    print(f"{len(questions)} questions x {args.runs} run(s), {calls} model calls")

    rows = []
    try:
        for question in questions:
            rows.append(score_question(question, runs=args.runs))
            last = rows[-1]
            score = "-" if last["faithfulness"] is None else f"{last['faithfulness']:.0%}"
            print(f"  {last['id']} {score}")
    except MissingApiKey as error:
        print(error)
        return

    summary = summarise(rows)
    print_rows(rows)
    print(
        f"\nmean faithfulness {summary['mean']:.1%} over {summary['claims']} claims; "
        f"{summary['perfect']}/{summary['scored']} answers fully supported"
    )

    if args.md:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(render_markdown(rows, summary), encoding="utf-8")
        print(f"\nWritten to {REPORT_PATH}")


if __name__ == "__main__":
    main()
