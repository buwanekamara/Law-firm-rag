"""Masking personal data before it leaves the machine.

Everything else runs locally; the gateway call is the one moment contract text
crosses the network, so it is the one place worth masking. Names and contact
details go out as stable placeholders and are restored in the answer.

What is not masked matters more. Over these five contracts Presidio flags 145
dates, 122 locations and 44 people - including "Heritage", a contracting
party. Masking dates loses "when does this take effect", locations loses
"which state's law governs", and "Heritage" makes its contract unanswerable.

So this is narrow: contact details and real person names above a confidence
threshold, never a term the contract defines. Contracts mark those with
quotation marks - ("Heritage"), ("Transporter") - which gives an allowlist
taken from the corpus instead of hand-written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from app.config import settings

# Excludes DATE_TIME, LOCATION, ORG, NRP and URL - see the module docstring.
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

# Presidio scores each detection. In this corpus a ZIP+4 read as a social
# security number scores 0.05 and real detections score 0.85 or above, so the
# threshold has room either side.
MIN_CONFIDENCE = settings.masking_min_confidence

# Longer than this, or containing a newline, is a parsing artefact rather than
# a name - "Schedule B.\n(iii) Heritage" arrives as a single PERSON.
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

    Parties, facilities, brands - masking these would make the text
    unanswerable. Read from the chunks, so a new corpus needs no new code.
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

    `mapping` carries across calls so one person is PERSON_1 in every excerpt
    of a request; otherwise the model cannot tell two placeholders apart.
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

    Longest first, so PERSON_1 cannot match the front of PERSON_10.
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
