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
LLM_MODEL). A model asked to grade its own output tends to agree with itself;
using a second model removes the most obvious form of that bias, though not
all of it - two models trained on overlapping data can share a blind spot, and
this is a check, not a proof.

    uv run eval/faithfulness_eval.py
    uv run eval/faithfulness_eval.py --md
    uv run eval/faithfulness_eval.py --ids t01 t04
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from app.answer import answer_question, extract_json
from app.config import PROJECT_ROOT, settings
from app.indexing import collection_size
from app.llm import MissingApiKey, complete
from app.prompting import render

QUESTIONS_PATH = Path(__file__).with_name("questions.jsonl")
REPORT_PATH = PROJECT_ROOT / "docs" / "faithfulness-eval.md"

JUDGE_PROMPT = "judge_v1"


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


def score_question(question: dict) -> dict:
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
    print(f"\n{'id':<6}{'claims':>8}{'supported':>11}{'faithful':>10}  unsupported claims")
    print("-" * 104)
    for row in rows:
        score = "-" if row["faithfulness"] is None else f"{row['faithfulness']:.0%}"
        bad = [c["claim"] for c in row["claims"] if not c.get("supported")]
        summary = bad[0][:44] + ("..." if len(bad[0]) > 44 else "") if bad else row["note"]
        print(f"{row['id']:<6}{row['total']:>8}{row['supported']:>11}{score:>10}  {summary}")


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
    }


def render_markdown(rows: list[dict], summary: dict) -> str:
    lines = [
        "# Faithfulness evaluation",
        "",
        f"Generated by `uv run eval/faithfulness_eval.py --md` on {date.today().isoformat()}.",
        "",
        f"Answers from `{settings.llm_model}` (prompt {settings.prompt_version}), judged by "
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
    parser.add_argument("--md", action="store_true", help="write docs/faithfulness-eval.md")
    args = parser.parse_args()

    if collection_size() == 0:
        print("The collection is empty. Run: uv run scripts/index.py")
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
    print(f"{len(questions)} questions, two model calls each")

    rows = []
    try:
        for question in questions:
            rows.append(score_question(question))
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
