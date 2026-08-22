# Results snapshot

One page gathering every measured number, taken **21 August 2026**. Retrieval
figures cover the full 24-question set; the answer and faithfulness figures
were taken before the conversational interface was added and are re-run as the
final acceptance step. Each figure links to
the file that produced it and the command that regenerates it.

Configuration for all runs: answers from `openai/gpt-4o-mini` at temperature 0,
judged by `anthropic/claude-haiku-4.5`, hybrid retrieval, `TOP_K=8`,
`MIN_SCORE=0.56`, 120 chunks from 5 contracts.

---

## Corpus

| | |
|---|---|
| Contracts | 5 |
| Pages | 47 |
| Chunks | 120 |
| Words indexed | 20,579 |
| Median chunk | 145 words |
| Redaction markers preserved | 94 |
| Placeholder dates preserved | 9 |

Detail: [chunk-inventory.md](chunk-inventory.md) — `uv run scripts/chunk.py --md`

---

## Retrieval

21 questions with a known correct section. Two more - the not-in-corpus
question and the general-terminology question - are reported but excluded from
scoring, since nothing can be retrieved correctly for either. Top-8 retrieved.

| Mode | hit@1 | hit@3 | hit@5 | MRR |
|---|---:|---:|---:|---:|
| **hybrid (dense + BM25, RRF)** | 81% | 95% | **100%** | 0.89 |
| dense only | 86% | 90% | 90% | 0.88 |
| sparse only (BM25) | 81% | 90% | 95% | 0.87 |

Hybrid is the only mode that puts a correct section in the top five for every
scored question. The decisive rows are the two trap questions: asked for a
redacted price and a placeholder date, **dense retrieval misses both clauses
entirely** while BM25 finds them at ranks 3 and 1. In the other direction, BM25
misses the question a dense search answers at rank 1. Neither half is reliable
alone.

An earlier evaluation on 17 easier questions showed hybrid *behind* dense and
led to dense being made the default. Adding the trap questions reversed that
decision. Both rounds are reported.

Detail: [retrieval-eval.md](retrieval-eval.md) — `uv run eval/retrieval_eval.py --md`

---

## Relevance gate calibration

Cosine similarity of the closest chunk, measured across the golden set and five
off-topic control queries.

| | similarity |
|---|---|
| Genuine questions | 0.63 – 0.89 |
| Off-topic controls (baking, weather, Python) | 0.47 – 0.55 |
| **"Notice period for the office lease" (not in corpus)** | **0.668** |

The threshold is set to **0.56**, which rejects queries that are not about
these contracts. It cannot reject a *contract* question the corpus cannot
answer: the office-lease question scores above two genuine questions, because
similarity measures topic, not answerability. That job is done by the prompt
rules and citation verification instead.

Regenerate: `uv run scripts/calibrate_gate.py`

---

## Answer quality and trap handling

22 questions, 3 runs each, majority vote. Deterministic checks only — cited
section, and the language used for redactions, placeholders and refusals.

| Prompt | Passed | Traps | Citations rejected by the guard | Unstable across runs |
|---|---:|---:|---:|---:|
| v1 baseline | 17/22 | 1/5 | 1 | 0 |
| v2 + redaction and placeholder rules | 19/22 | 2/5 | 0 | 0 |
| v3 + cross-reference rule, cite-when-refusing | 21/22 | 4/5 | 0 | 1 |
| **v4 + name the contract in the prose** | **21/22** | **4/5** | **0** | **0** |

The five traps are the questions whose honest answer is "the contracts do not
give you this", in four disguises: a redacted value `[***]`, a blank
placeholder `[·]`, a subject absent from the corpus, and a clause that defers
its substance to a document not provided.

v3 and v4 tie on score but fail different questions; v4 ships for stability and
because naming the contract in the prose is required by the interface.

Detail: [answer-eval.md](answer-eval.md) —
`uv run eval/answer_eval.py --versions v1 v2 v3 v4 --runs 3 --md`

---

## Faithfulness

Each answer decomposed into atomic claims by a second model, each claim checked
against the excerpts that answer was built from.

| | |
|---|---|
| Questions scored | 22/22 |
| Claims checked | 79 |
| Claims supported | 79 |
| Mean faithfulness | **100%** |

Detail: [faithfulness-eval.md](faithfulness-eval.md) — `uv run eval/faithfulness_eval.py --md`

A perfect score deserves scepticism rather than celebration, and two caveats
belong with it. Faithfulness measures whether an answer stayed inside its
sources — not whether it is correct in the world, and not whether it is
complete: an answer that faithfully reports one clause while missing a second
relevant one still scores 100%. And the judge is itself a language model.

---

## Tests

133 tests, no network and no Docker required — the embedding models are
replaced with deterministic stand-ins and Qdrant runs in-process.

Several tests pin behaviour that was broken and fixed, so the failure message
names what regressed: per-document heading detection, the appendix boundary,
citation matching, and negation phrasing.

`uv run pytest`

---

## Known limitations

- **t04** (gas quality) fails under v4: the answer restates a cross-reference
  instead of saying the specification is not set out. One borderline question
  of 22, in the hardest category. Prompt work stopped here deliberately — a
  fifth version tuned against a 22-question set would be fitting the test.
- The relevance gate cannot distinguish an unanswerable contract question from
  an answerable one, as calibration showed.
- Faithfulness does not measure completeness.
- Five bugs were found in the evaluation harness itself during this work; each
  was fixed and every prompt version re-scored afterwards. Three scoring rules
  changed after a result was surprising.
