"""Phase 4 - orchestration: retrieve, prompt, parse.

The flow is deliberately linear and inspectable:

    question -> retrieve chunks -> render prompt -> model -> parsed answer

Every stage is available for inspection through the `debug` flag, because the
next two phases consist almost entirely of looking at exactly what went into
the prompt and exactly what came back.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.chunking import citation_header
from app.config import settings
from app.guards import verify_citations
from app.llm import complete
from app.masking import describe, mask_excerpts, unmask_text
from app.prompting import load_prompt, render_user_prompt
from app.retrieval import SearchResult, best_similarity, search

# Models sometimes wrap JSON in a code fence despite being told not to.
_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

NOTHING_RETRIEVED = "No relevant clause was found in the contracts provided."
BELOW_THRESHOLD = (
    "No sufficiently relevant clause was found in the contracts provided, so no answer "
    "was generated."
)


class Citation(BaseModel):
    """A citation as the model reports it - not yet verified.

    Phase 6 checks these against the chunks that were actually retrieved. At
    this stage they are only as trustworthy as the model.
    """

    doc_title: str = ""
    section: str = ""
    page: int | None = None


class ModelAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: str = "medium"


@dataclass
class AnswerResult:
    question: str
    answer: str
    citations: list[dict[str, Any]]
    confidence: str
    retrieved: list[dict[str, Any]]
    doc_filter: str | None = None
    parse_error: str | None = None
    # Citations the model claimed that do not correspond to any chunk it was
    # shown. They are removed from `citations` rather than displayed, because
    # a citation is a promise a reader can check.
    rejected_citations: list[dict[str, Any]] = field(default_factory=list)
    gated: bool = False
    gate_score: float | None = None
    warnings: list[str] = field(default_factory=list)
    debug: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.debug is None:
            payload.pop("debug")
        return payload


def build_excerpts(results: list[SearchResult]) -> str:
    """Render retrieved chunks for the prompt, each stamped with its source.

    The numbering gives the model a short handle for each excerpt, and the
    header carries the exact document, section and page that a citation must
    reproduce - so a correct citation is a copy, not a recollection.
    """
    blocks = []
    for result in results:
        blocks.append(f"[{result.rank}] {citation_header(result.chunk)}\n{result.chunk['text']}")
    return "\n\n".join(blocks)


def extract_json(raw: str) -> dict[str, Any]:
    """Pull the JSON object out of a model reply.

    Deliberately forgiving: strip code fences, then take everything between
    the first { and the last }. Anything a model bolts on before or after the
    object is discarded rather than causing a failure.
    """
    text = _CODE_FENCE.sub("", raw).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in the reply")
    return json.loads(text[start : end + 1])


def parse_answer(raw: str) -> ModelAnswer:
    return ModelAnswer.model_validate(extract_json(raw))


def answer_question(
    question: str,
    top_k: int | None = None,
    doc_id: str | None = None,
    debug: bool = False,
    prompt_version: str | None = None,
    client: Any = None,
) -> AnswerResult:
    """Answer one question from the indexed contracts."""
    top_k = top_k or settings.top_k
    results = search(question, top_k=top_k, doc_id=doc_id, client=client)

    retrieved = [
        {
            "rank": result.rank,
            "score": result.score,
            "chunk_id": result.chunk["chunk_id"],
            **result.citation,
        }
        for result in results
    ]

    if not results:
        return AnswerResult(
            question=question,
            answer=NOTHING_RETRIEVED,
            citations=[],
            confidence="low",
            retrieved=[],
            doc_filter=doc_id,
        )

    # Guard one: refuse before the model is involved.
    #
    # Instructions in a prompt are a request; this is a gate. If nothing in
    # the corpus is close to the question, the model never gets the chance to
    # assemble a plausible answer from unrelated clauses - which is the
    # failure mode that produces the most confident nonsense.
    gate_score = None
    if settings.min_score > 0:
        gate_score = best_similarity(question, doc_id=doc_id, client=client)
        if gate_score < settings.min_score:
            return AnswerResult(
                question=question,
                answer=BELOW_THRESHOLD,
                citations=[],
                confidence="low",
                retrieved=retrieved,
                doc_filter=doc_id,
                gated=True,
                gate_score=gate_score,
                warnings=[
                    f"best similarity {gate_score:.3f} is below the threshold "
                    f"{settings.min_score:.3f}; no model call was made"
                ],
            )

    system_prompt, _ = load_prompt(prompt_version)

    # Guard three: mask personal data before the text crosses the network.
    # Everything up to here ran locally; this is the only moment contract text
    # leaves the machine.
    masked = mask_excerpts(build_excerpts(results))
    user_prompt = render_user_prompt(question, masked.text, prompt_version)

    raw = complete(system_prompt, user_prompt)

    parse_error = None
    try:
        parsed = parse_answer(raw)
    except (ValueError, ValidationError, json.JSONDecodeError) as first_error:
        # One corrective attempt. If a model cannot produce the object twice
        # in a row, the raw text is returned rather than an exception - a
        # readable answer with a flag on it beats a 500.
        retry_prompt = (
            f"{user_prompt}\n\nYour previous reply could not be parsed as JSON "
            f"({first_error}). Reply with the JSON object only."
        )
        # The retry's reply becomes the reported one either way, so `debug`
        # always shows the last thing the model actually said.
        raw = complete(system_prompt, retry_prompt)
        try:
            parsed = parse_answer(raw)
        except (ValueError, ValidationError, json.JSONDecodeError) as second_error:
            parse_error = str(second_error)
            parsed = ModelAnswer(answer=raw, citations=[], confidence="low")

    # Guard two: every citation must name a chunk the model was actually
    # shown. Cheap, deterministic, and it catches the failure that matters
    # most here - an answer that looks properly sourced but points at a
    # section nobody put in front of it.
    shown = [{"doc_title": item["doc_title"], "section_label": item["section"]} for item in retrieved]
    claimed = [citation.model_dump() for citation in parsed.citations]
    supported, unsupported = verify_citations(claimed, shown)

    if masked.mapping:
        # The model reasoned about placeholders; the reader sees real names.
        parsed.answer = unmask_text(parsed.answer, masked.mapping)
        for citation in parsed.citations:
            citation.doc_title = unmask_text(citation.doc_title, masked.mapping)
            citation.section = unmask_text(citation.section, masked.mapping)

    warnings: list[str] = []
    if settings.masking_enabled:
        warnings.append(f"PII masking on: {describe(masked)}")

    if unsupported and settings.citation_retry:
        allowed = ", ".join(sorted({item["section_label"] for item in shown}))
        invented = ", ".join(c.get("section", "?") for c in unsupported)
        correction = (
            f"{user_prompt}\n\nYour previous reply cited {invented}, which is not among the "
            f"excerpts you were given. Cite only these sections: {allowed}. "
            "Answer again, using the same JSON format."
        )
        try:
            reparsed = parse_answer(complete(system_prompt, correction))
        except (ValueError, ValidationError, json.JSONDecodeError):
            reparsed = None
        if reparsed is not None:
            parsed = reparsed
            claimed = [citation.model_dump() for citation in parsed.citations]
            supported, unsupported = verify_citations(claimed, shown)
            warnings.append("regenerated once after an unsupported citation")

    if unsupported:
        invented = ", ".join(
            f"{c.get('doc_title', '?')} {c.get('section', '?')}" for c in unsupported
        )
        warnings.append(f"removed unsupported citation(s): {invented}")

    result = AnswerResult(
        question=question,
        answer=parsed.answer,
        citations=supported,
        confidence=parsed.confidence,
        retrieved=retrieved,
        doc_filter=doc_id,
        parse_error=parse_error,
        rejected_citations=unsupported,
        gate_score=gate_score,
        warnings=warnings,
    )

    if debug:
        result.debug = {
            "prompt_version": prompt_version or settings.prompt_version,
            "model": settings.llm_model,
            "search_mode": settings.search_mode,
            "system_prompt": system_prompt,
            # The prompt exactly as sent - masked, if masking is on.
            "user_prompt": user_prompt,
            "raw_response": raw,
            "masking": {
                "enabled": settings.masking_enabled,
                "entities": masked.entity_counts,
                "placeholders": sorted(masked.mapping),
            },
            "chunks": [
                {"rank": r.rank, "header": citation_header(r.chunk), "text": r.chunk["text"]}
                for r in results
            ],
        }
    return result
