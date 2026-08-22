"""The faithfulness harness and its self-test.

The judge itself needs a live model, so what is tested here is the harness
around it: that the controls are well formed, and that the scoring arithmetic
does what it claims.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from faithfulness_eval import (
    NEGATIVE_CONTROLS,
    POSITIVE_CONTROL,
    summarise,
)


def test_the_controls_cover_distinct_failure_modes():
    """Four different ways to be wrong, not four spellings of one."""
    names = {control["name"] for control in NEGATIVE_CONTROLS}
    assert names == {
        "invented figure",
        "right clause, wrong contract",
        "inference from silence",
        "plausible but contradicted",
    }


@pytest.mark.parametrize("control", NEGATIVE_CONTROLS, ids=lambda c: c["name"])
def test_each_control_states_what_was_planted(control):
    """The planted claim is written down so a miss can be read as a sentence
    rather than inferred from a score."""
    assert control["question"] and control["answer"]
    assert len(control["planted"].split()) >= 4


def test_a_positive_control_exists():
    """A judge that flags everything is as useless as one that flags nothing,
    so one wholly supported answer has to survive."""
    assert POSITIVE_CONTROL["answer"]
    assert "California" in POSITIVE_CONTROL["answer"]


def test_the_negative_controls_are_not_accidentally_true():
    """Each planted claim must contradict or exceed the contracts - not merely
    paraphrase them."""
    planted = " ".join(control["answer"] for control in NEGATIVE_CONTROLS)
    assert "4% per month" in planted          # no such fee exists
    assert "Hosting Agreement" in planted     # the clause belongs to another contract
    assert "thirty days" in planted           # the clause says immediately


# --- the arithmetic --------------------------------------------------------

def test_faithfulness_is_supported_over_total():
    rows = [
        {"faithfulness": 1.0, "supported": 4, "total": 4},
        {"faithfulness": 0.5, "supported": 1, "total": 2},
    ]
    summary = summarise(rows)
    assert summary["claims"] == 6
    assert summary["supported"] == 5
    assert summary["mean"] == pytest.approx(0.75)
    assert summary["perfect"] == 1


def test_unscored_rows_are_excluded_not_counted_as_zero():
    """A gated question has no claims to check. Counting it as 0% would blame
    the answer for a refusal that was correct."""
    rows = [
        {"faithfulness": 1.0, "supported": 3, "total": 3},
        {"faithfulness": None, "supported": 0, "total": 0},
    ]
    summary = summarise(rows)
    assert summary["scored"] == 1
    assert summary["mean"] == 1.0
