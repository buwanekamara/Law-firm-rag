"""Choose the refusal threshold from data rather than by guessing.

    uv run scripts/calibrate_gate.py

For every question in the golden set it prints the cosine similarity of the
closest chunk, splitting them into questions the corpus can answer and the one
it cannot. A usable threshold sits between the two groups: above the
not-in-corpus question, below the weakest real question.

If the groups overlap there is no safe threshold, and the honest response is
to leave gating off and say so - a gate that rejects real questions is worse
than no gate at all.

The score used is dense cosine similarity, not the fused hybrid score.
Reciprocal rank fusion ranks by agreement between two result lists, so the
top result of a nonsense query scores about as well as the top result of a
good one. See app/retrieval.best_similarity.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.search.indexing import collection_size
from app.search.retrieval import best_similarity

QUESTIONS_PATH = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"

# Negative controls that are not about contracts at all. The golden set's
# not_in_corpus question ("the office lease") is deliberately contract-shaped,
# so it tests whether similarity can tell absent-from-this-corpus from
# present. These test something weaker but still useful: whether it can tell
# a contract question from a question about anything else.
OFF_TOPIC_PROBES = (
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "Write me a Python function that reverses a string.",
    "zebra quantum bicycle recipe",
    "What is the weather tomorrow?",
)


def main() -> None:
    if collection_size() == 0:
        print("The collection is empty. Run: uv run scripts/index.py")
        return

    questions = [
        json.loads(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    answerable: list[tuple[str, float]] = []
    unanswerable: list[tuple[str, float]] = []
    print(f"{'id':<6}{'similarity':>12}  {'type':<16}question")
    print("-" * 100)
    for question in questions:
        score = best_similarity(question["question"])
        bucket = unanswerable if question["type"] == "not_in_corpus" else answerable
        bucket.append((question["id"], score))
        print(f"{question['id']:<6}{score:>12.4f}  {question['type']:<16}{question['question'][:56]}")

    print(f"\n{'probe':<6}{'similarity':>12}  off-topic control")
    print("-" * 100)
    off_topic = []
    for probe in OFF_TOPIC_PROBES:
        score = best_similarity(probe)
        off_topic.append(score)
        print(f"{'--':<6}{score:>12.4f}  {probe[:56]}")

    weakest_real = min(answerable, key=lambda pair: pair[1]) if answerable else ("-", 0.0)
    strongest_fake = max(unanswerable, key=lambda pair: pair[1]) if unanswerable else ("-", 0.0)
    strongest_off_topic = max(off_topic) if off_topic else 0.0

    print(f"\nweakest answerable question:      {weakest_real[0]} at {weakest_real[1]:.4f}")
    print(f"strongest unanswerable question:  {strongest_fake[0]} at {strongest_fake[1]:.4f}")
    print(f"strongest off-topic probe:        {strongest_off_topic:.4f}")

    print("\n--- can the gate reject a contract question this corpus cannot answer? ---")
    if strongest_fake[1] >= weakest_real[1]:
        print(
            "NO. The groups overlap, so any threshold that rejects it would also reject a\n"
            "real question. Similarity measures topic, not answerability, and a question\n"
            "about a lease is contract-shaped whether or not a lease exists here.\n"
            "That job belongs to the prompt's refusal rules and to citation verification."
        )
    else:
        print(f"Yes - a threshold between {strongest_fake[1]:.4f} and {weakest_real[1]:.4f} separates them.")

    print("\n--- can it reject a question that is not about contracts at all? ---")
    if strongest_off_topic >= weakest_real[1]:
        print("NO. Even unrelated questions score as highly as real ones. Leave MIN_SCORE=0.")
        return

    suggested = strongest_off_topic + (weakest_real[1] - strongest_off_topic) / 3
    print(
        f"Yes. Off-topic tops out at {strongest_off_topic:.4f}; the weakest real question is\n"
        f"{weakest_real[1]:.4f}. A threshold in between rejects nonsense without touching\n"
        "anything genuine.\n"
        f"\nSuggested MIN_SCORE={suggested:.3f}  (a third of the way up the gap - deliberately\n"
        "closer to the off-topic end, because refusing a real question is the worse error)."
    )
    print(f"Currently configured: MIN_SCORE={settings.min_score}")
    print(
        "\nBe clear about what this buys: it filters queries that are not about these\n"
        "contracts. It does NOT decide whether the corpus can answer a contract question."
    )


if __name__ == "__main__":
    main()
