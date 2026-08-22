"""Phase 6.5 - masking personal data before it leaves the machine.

Retrieval, embedding and chunking all happen locally. The one moment contract
text crosses the network is the call to the model gateway, so that is the one
place worth masking. Person names and contact details are replaced with stable
placeholders on the way out and restored in the answer on the way back, so the
model reasons about "PERSON_1" and the reader sees the real name.

What is *not* masked matters more than what is, and this corpus makes the point
sharply. Presidio, run over these five contracts, flags:

  DATE_TIME  145 hits   "January 11, 2018", "12 weeks", "9 a.m. to 5 p.m."
  LOCATION   122 hits   "Delaware", "Santa Ana", "Pennsylvania"
  PERSON      44 hits   including "Jasper" and "Heritage"
  US_SSN       1 hit    "92703-1310" - a ZIP+4, at 0.05 confidence

Masking dates would remove the answer to "when does this take effect". Masking
locations would remove the answer to "which state's law governs". And masking
"Heritage" - a contracting party, not a person - would make the manufacturing
agreement unanswerable. None of those are personal data in a contract filed
publicly with a securities regulator.

So masking here is deliberately narrow: contact details and genuine person
names only, above a confidence threshold, and never a term the contract itself
defines. Contracts announce their load-bearing names by putting them in
quotation marks - ("Heritage"), ("Transporter"), ("Brand") - which gives an
allowlist derived from the corpus rather than hand-written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from app.config import settings

# Entities worth masking. Deliberately excludes DATE_TIME, LOCATION, ORG, NRP
# and URL - see the module docstring.
MASKED_ENTITIES = (
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_BANK_NUMBER",
    "US_PASSPORT",
)

# Presidio reports a confidence per detection. The ZIP+4 mistaken for a social
# security number scored 0.05; genuine detections in this corpus score 0.85 or
# 1.0, so the threshold has plenty of room.
MIN_CONFIDENCE = settings.masking_min_confidence

# A detected span longer than this, or containing a line break, is a parsing
# artefact rather than a name. Presidio produced "Schedule B.\n(iii) Heritage"
# as a single PERSON.
MAX_SPAN_LENGTH = 40

# Defined terms: ("Heritage"), ("Transporter"). Curly or straight quotes.
_DEFINED_TERM = re.compile(r"[“\"']([A-Z][\w &.,'-]{1,40})[”\"']")

_PLACEHOLDER = re.compile(r"\b([A-Z_]+)_(\d+)\b")


class MaskingUnavailable(RuntimeError):
    """Presidio is not installed but masking was switched on."""


@dataclass
class MaskResult:
    text: str
    # placeholder -> the original string it stands for
    mapping: dict[str, str] = field(default_factory=dict)
    entity_counts: dict[str, int] = field(default_factory=dict)

    @property
    def masked_count(self) -> int:
        return sum(self.entity_counts.values())


@lru_cache(maxsize=1)
def _analyzer():
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise MaskingUnavailable(
            "MASKING_ENABLED is true but presidio is not installed. Run:\n"
            "  uv add presidio-analyzer presidio-anonymizer\n"
            "  uv run python -m spacy download en_core_web_lg"
        ) from error
    return AnalyzerEngine()


@lru_cache(maxsize=1)
def defined_terms() -> frozenset[str]:
    """Every term the corpus defines in quotation marks.

    These are the names the contracts run on - parties, facilities, brands -
    and replacing them with placeholders would make the text unanswerable.
    Derived from the chunks rather than hand-listed, so a new corpus needs no
    new code.
    """
    try:
        from app.ingest.chunking import load_chunks

        chunks = load_chunks()
    except (FileNotFoundError, OSError):
        return frozenset()

    terms: set[str] = set()
    for chunk in chunks:
        for match in _DEFINED_TERM.finditer(chunk["text"]):
            term = match.group(1).strip()
            terms.add(term)
            # ("Jasper Products, L.L.C.") also protects "Jasper"
            for word in term.split():
                if len(word) > 3 and word[0].isupper():
                    terms.add(word.strip(".,"))
    return frozenset(terms)


def _is_protected(value: str) -> bool:
    protected = defined_terms()
    if value in protected:
        return True
    return any(word.strip(".,") in protected for word in value.split())


def mask_text(text: str, mapping: dict[str, str] | None = None) -> MaskResult:
    """Replace personal data with stable placeholders.

    `mapping` carries state between calls, so the same person is PERSON_1 in
    every excerpt of one request - otherwise the model sees two placeholders
    and cannot tell they are the same party.
    """
    mapping = dict(mapping or {})
    reverse = {original: placeholder for placeholder, original in mapping.items()}
    counts: dict[str, int] = {}

    results = [
        r
        for r in _analyzer().analyze(text=text, language="en", entities=list(MASKED_ENTITIES))
        if r.score >= MIN_CONFIDENCE
    ]
    # Replace from the end so earlier offsets stay valid.
    for result in sorted(results, key=lambda r: r.start, reverse=True):
        value = text[result.start : result.end]
        if len(value) > MAX_SPAN_LENGTH or "\n" in value or _is_protected(value):
            continue

        placeholder = reverse.get(value)
        if placeholder is None:
            index = sum(1 for key in mapping if key.startswith(result.entity_type)) + 1
            placeholder = f"{result.entity_type}_{index}"
            mapping[placeholder] = value
            reverse[value] = placeholder
        counts[result.entity_type] = counts.get(result.entity_type, 0) + 1
        text = text[: result.start] + placeholder + text[result.end :]

    return MaskResult(text=text, mapping=mapping, entity_counts=counts)


def unmask_text(text: str, mapping: dict[str, str]) -> str:
    """Put the real values back.

    Longest placeholders first, so PERSON_10 is restored before PERSON_1 can
    match the start of it.
    """
    for placeholder in sorted(mapping, key=len, reverse=True):
        text = text.replace(placeholder, mapping[placeholder])
    return text


def mask_excerpts(text: str) -> MaskResult:
    """Mask a whole rendered excerpt block, if masking is switched on."""
    if not settings.masking_enabled:
        return MaskResult(text=text)
    return mask_text(text)


def describe(result: MaskResult) -> str:
    if not result.entity_counts:
        return "nothing masked"
    parts = ", ".join(f"{count} x {entity}" for entity, count in sorted(result.entity_counts.items()))
    return f"masked {parts}"
