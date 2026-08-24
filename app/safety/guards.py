"""Checks that sit around the model rather than inside the prompt.

An instruction is a request and can be declined. A check is code, and runs
every time. Used both by the evaluation and by the answer path, so an invented
citation is caught before anyone reads it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

# "Section 4.3 - Termination for Breach" -> "4.3"
_LABEL_PREFIX = re.compile(r"^\s*(section|article|exhibit|schedule|clause)\s+", re.IGNORECASE)
_LABEL_SUFFIX = re.compile(r"\s*[-–—:|].*$")  # noqa: RUF001 - en dashes appear in section labels

# One vocabulary for "the contracts do not answer this", shared by the refusal
# and cross-reference checks so both agree on what counts.
#
# A regex rather than literal strings, because models insert adverbs: "does
# not explicitly state" contains neither "does not state" nor "not stated". Up
# to two words are allowed between the negation and the verb.
_NOT_ANSWERED = re.compile(
    r"""
      do(?:es)?\s+not\s+(?:\w+\s+){0,2}
        (?:contain|address|specify|state|mention|set\s+out|define|provide|deal\s+with|cover)
    | (?:is|are|was|were)\s+not\s+(?:\w+\s+){0,2}
        (?:specified|stated|set\s+out|defined|addressed|mentioned|provided|contained|covered)
    | \bnot\s+(?:\w+\s+){0,2}
        (?:specified|stated|set\s+out|defined|addressed|mentioned)\b
    | \bno\s+(?:relevant|information|provision|clause|mention|details?)\b
    | cannot\s+be\s+answered
    | \bsilent\s+on\b
    | unclear\s+based\s+on
    | not\s+found\s+in
    | outside\s+the\s+(?:scope|excerpts|provided)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# The bare markers are excluded on purpose. An answer that only echoes "[.]"
# reads as though a date were given. What matters is whether the marker gets
# explained, so the explanation is what these match.
REDACTION_PATTERNS = ("redact", "withheld", "omitted", "not disclosed")
PLACEHOLDER_PATTERNS = (
    "placeholder", "left blank", "not filled", "blank in", "unfilled", "never completed",
    "was not completed", "left incomplete",
)
# A clause that addresses a topic without stating the value sounds like a
# refusal; what separates them is whether a section was cited.


def parse_label(label: str) -> tuple[str, str]:
    """Split a section label into (kind, identifier), both casefolded.

    The model writes "4.2 - Termination for Convenience", bare "4.2" and
    "Section 4.2" interchangeably, so comparing raw strings marks honest
    citations as invented.

    The kind is kept because one contract has both an Article X and an Exhibit
    A. When either side omits it, the identifier alone decides.
    """
    if not label:
        return "", ""
    text = _LABEL_SUFFIX.sub("", str(label).strip())
    match = _LABEL_PREFIX.match(text)
    kind = match.group(1).casefold() if match else ""
    identifier = _LABEL_PREFIX.sub("", text).strip().rstrip(".").casefold()
    return kind, identifier


def normalise_label(label: str) -> str:
    """The identifier alone - useful for display and for loose comparisons."""
    return parse_label(label)[1]


def normalise_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(title or "").casefold()).strip()


def citation_matches(citation: dict[str, Any], chunk: dict[str, Any]) -> bool:
    """Does a claimed citation correspond to a chunk that was retrieved?"""
    claimed_kind, claimed_id = parse_label(citation.get("section", ""))
    actual_kind, actual_id = parse_label(chunk.get("section", chunk.get("section_label", "")))
    if not claimed_id or claimed_id != actual_id:
        return False
    if claimed_kind and actual_kind and claimed_kind != actual_kind:
        return False
    claimed_title = normalise_title(citation.get("doc_title", ""))
    actual_title = normalise_title(chunk.get("doc_title", ""))
    # A missing title on either side is tolerated - the section label is the
    # load-bearing part. It has to be tolerated on both sides, or every
    # citation naming its document fails against a target that does not.
    if not claimed_title or not actual_title:
        return True
    return claimed_title == actual_title


def verify_citations(
    citations: Iterable[dict[str, Any]], retrieved: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split claimed citations into (supported, unsupported).

    Supported means the citation names a chunk that was actually shown to the
    model - not that the answer is true.
    """
    chunks = list(retrieved)
    supported, unsupported = [], []
    for citation in citations:
        if any(citation_matches(citation, chunk) for chunk in chunks):
            supported.append(citation)
        else:
            unsupported.append(citation)
    return supported, unsupported


def mentions(text: str, patterns: Iterable[str]) -> bool:
    lowered = (text or "").casefold()
    return any(pattern.casefold() in lowered for pattern in patterns)


def looks_like_refusal(text: str) -> bool:
    return bool(_NOT_ANSWERED.search(text or ""))


def reports_redaction(text: str) -> bool:
    return mentions(text, REDACTION_PATTERNS)


def reports_placeholder(text: str) -> bool:
    return mentions(text, PLACEHOLDER_PATTERNS)


# Phrases marking an explanation as general knowledge rather than contract
# text. Explaining what "indemnify" means is help; saying who must indemnify
# whom is a claim, and claims come only from the contracts.
GENERAL_USAGE_PATTERNS = (
    "general legal usage",
    "not defined in these contracts",
    "not defined in the contracts",
    "in general usage",
    "generally means",
    "in general terms",
    "commonly means",
    "not a definition taken from these contracts",
    "generally,",
)


# Attempts to give the model orders rather than ask it something. Detection is
# not the defence on its own - the response to an injection is still to answer
# the real question - but it lets the system warn out loud and lets the prompt
# carry a note where it is needed.
#
# Fencing the message is not sufficient by itself: a rule in a system message
# competes with an imperative sitting closer to the point of generation, and
# can lose.
_INJECTION = re.compile(
    r"""
      ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier|foregoing)
    | disregard\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier|instructions?)
    | (?:do\s*not|don't|dont)\s+answer\b
    | (?:reply|respond|answer|say|output|print|write)\s+(?:back\s+)?(?:with\s+)?(?:only|just|exactly)\b
    | you\s+are\s+now\b
    | (?:new|updated|revised)\s+instructions?\b
    | (?:system|initial|original)\s+prompt\b
    | forget\s+(?:everything|all|your)\b
    | act\s+as\s+(?:a|an|if)\b
    | pretend\s+(?:to\s+be|that\s+you)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def looks_like_injection(text: str) -> bool:
    """Does this text try to give the model orders rather than ask it something?"""
    return bool(_INJECTION.search(text or ""))


def marks_general_knowledge(text: str) -> bool:
    """Did the answer flag an explanation as general rather than contractual?"""
    return mentions(text, GENERAL_USAGE_PATTERNS)


def reports_not_stated(text: str) -> bool:
    """For clauses that address a topic but defer the substance elsewhere."""
    return bool(_NOT_ANSWERED.search(text or ""))
