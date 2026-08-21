# Development Blueprint — Contract-RAG Prototype

FastAPI + uv, built in small, independently debuggable steps. Each phase produces something you can run and inspect on its own before the next phase depends on it. A UI comes last (optional) — until then, FastAPI's auto-generated `/docs` page *is* the demo UI.

This extends the scope doc (§1–§3): Qdrant, fastembed (`bge-small-en-v1.5`), PyMuPDF, hybrid dense+BM25 with RRF, Presidio masking, Vercel AI Gateway.

---

## 0. Project scaffold (½ hour)

**Goal:** a runnable skeleton where every later phase has a home and a debug entry point.

Work directly in the existing project folder (`D:\MyProjects\dxdy`) — no new folder:

```
cd D:\MyProjects\dxdy
uv init . --python 3.12   # then rename to contract-rag in pyproject.toml
uv add fastapi "uvicorn[standard]" pydantic-settings pymupdf fastembed qdrant-client openai
uv add --dev pytest ruff
```

Layout — one module per pipeline stage, one debug CLI per stage. The `contracts/` folder already there stays as-is and serves as the raw input directory:

```
dxdy/
├── contracts/                # the 5 contract PDFs (already present — raw input)
├── Take Home Assessment ….pdf# the brief (keep out of the submission zip)
├── pyproject.toml            # deps + [project.scripts] entry points
├── uv.lock                   # commit this; NEVER commit .env
├── .env.example              # AI_GATEWAY_API_KEY=  LLM_MODEL=  JUDGE_MODEL=
├── .gitignore                # .env, .venv, data/ outputs, __pycache__
├── data/
│   ├── extracted/            # phase 1 output (JSON, inspectable)
│   └── chunks/               # phase 2 output (JSON, inspectable)
├── app/                      # the package (generic name, not the company's)
│   ├── config.py             # pydantic-settings: reads .env, single source of truth
│   ├── extraction.py         # phase 1
│   ├── chunking.py           # phase 2
│   ├── indexing.py           # phase 3 (embed + upsert to Qdrant)
│   ├── retrieval.py          # phase 3 (hybrid query + RRF)
│   ├── llm.py                # phase 4 (gateway client)
│   ├── prompts/              # phase 5 (versioned prompt files, jinja/f-string)
│   ├── guards.py             # phase 6 (grounding checks, citation verification)
│   ├── masking.py            # phase 6.5 (Presidio)
│   ├── answer.py             # orchestrates retrieve → mask → LLM → verify
│   └── api.py                # FastAPI app
├── scripts/                  # debug CLIs: extract.py, chunk.py, search.py, ask.py
├── eval/
│   ├── questions.jsonl       # golden Q/A set incl. trap questions
│   └── run_eval.py           # retrieval metrics + faithfulness judge
└── tests/
```

**Why per-stage debug CLIs:** when an answer is wrong you need to know *which* stage broke — extraction, chunking, retrieval, or generation. `uv run scripts/search.py "termination notice"` tells you in seconds whether retrieval is the problem before the LLM is even involved.

**Done when:** `uv run uvicorn app.api:app` serves a `/health` endpoint.

---

## 1. PDF extractor

**Goal:** clean per-page text with page numbers preserved, saved as JSON you can eyeball.

Steps:

1. PyMuPDF: per page, `page.get_text()` → `{doc_id, title, pages:[{page_no, text}]}` → `data/extracted/<doc>.json`.
2. Cleaning pass (do it here, not in chunking): fix hyphenation across line breaks, collapse whitespace, normalise non-breaking/typographic spaces (the transportation and joint-venture PDFs are built almost entirely from `\xa0`), strip the EDGAR `Source: ...` footer stamped on every page and page-number-only lines. (Checked against the real text: the "14t h day" broken-kerning problem does **not** occur in this corpus — PyMuPDF returns "14th day" correctly.)
3. **Preserve, don't clean:** `[***]` redaction markers and `[·]` placeholder dates must survive extraction — they are your hallucination test material later.

Debug: `uv run scripts/extract.py` prints per-doc char counts and dumps `.txt` files; diff-read one contract against the actual PDF. Tests: assert known strings appear ("Joint Venture", "[***]", page counts match pdfinfo).

**Done when:** all 5 docs extract with sane text, redaction markers intact.

---

## 2. Chunking (clause-aware)

**Goal:** chunks that map 1:1 to contract sections, because the assessment requires citing "the specific section" — this metadata is where citations come from.

Steps:

1. Regex-detect section boundaries: `^\d+\.` , `^Section \d+(\.\d+)*`, `^ARTICLE [IVX]+`, `WHEREAS`, `Exhibit` blocks. Contracts differ — tune per pattern, keep a fallback.
2. Emit chunks with metadata: `{chunk_id, doc_id, doc_title, section_id, section_heading, page_start, page_end, text}`.
3. Long sections: split at ~400–500 tokens with ~50-token overlap, but child chunks keep the parent's `section_id`. Never merge across section boundaries.
4. Save to `data/chunks/*.json`.

Debug: `uv run scripts/chunk.py --doc hosting` prints a table (section_id, heading, tokens, page). Sanity: histogram of chunk sizes; any 3000-token chunk means a missed boundary. Tests: "Section 1 of the Hosting Agreement is one chunk titled 'Website Design and Development'".

**Done when:** every chunk carries a correct section id + page range you could hand to a lawyer.

---

## 3. Retrieval (Qdrant, hybrid dense + BM25, RRF)

**Goal:** given a question, return the right chunks with scores — verifiable before any LLM exists.

Steps:

1. Run Qdrant alone: `docker run -p 6333:6333 -v ./qdrant_data:/qdrant/storage qdrant/qdrant` (Compose comes in phase 7).
2. `indexing.py`: one Qdrant collection with **two named vectors per point** — dense (`bge-small-en-v1.5` via fastembed, local CPU) and sparse (BM25 via fastembed). Chunk metadata goes in the payload.
3. `retrieval.py`: Qdrant Query API `prefetch` (dense + sparse) fused server-side with **RRF** (reciprocal rank fusion — each result's final score comes from its *rank* in each list, `Σ 1/(60+rank)`, so dense and sparse scores never need to be on the same scale).
4. Optional metadata filter: if the question names a contract ("in the hosting agreement…"), filter by `doc_id`.

Debug: `uv run scripts/search.py "confidentiality obligations" --k 5` → table of score, doc, section, first 80 chars. Mini retrieval eval now, before LLM: 10 questions with known correct sections in `eval/questions.jsonl`; measure hit@5. This isolates retrieval quality forever.

**Done when:** hit@5 ≥ ~8/10 on the golden set; defined-term queries ("Transporter", "the Brand") hit the right doc thanks to BM25.

---

## 4. LLM integration

**Goal:** end-to-end answer, plainest possible prompt. Correct plumbing, not clever prompting.

Steps:

1. `llm.py`: `openai` client with `base_url="https://ai-gateway.vercel.sh/v1"`, key from env. `LLM_MODEL` env-configurable. `temperature=0` from day one (deterministic = debuggable).
2. `answer.py`: question → retrieve top-k → stuff chunks into prompt with their metadata headers `[Hosting Agreement | Section 4 | p.2]` → ask.
3. Structured output (pydantic): `{answer, citations:[{doc, section, page}], confidence}`.
4. `api.py`: `POST /ask {question, top_k?}` → that JSON. Also `uv run scripts/ask.py "..."` for terminal use.

Debug: a `?debug=true` flag returning the retrieved chunks + the exact rendered prompt alongside the answer — you'll use this constantly in phases 5–6.

**Done when:** `curl -X POST /ask` returns a grounded answer with at least the right document cited.

---

## 5. Prompt optimization

**Goal:** tone, citation discipline, and refusal behavior — measured, not vibed.

Steps:

1. Prompts live as versioned files in `app/prompts/` (e.g. `answer_v1.md`, `answer_v2.md`), selected by env/param — so the report can show a before/after.
2. System prompt rules: professional and objective; **only** use provided excerpts; cite section per claim; if the excerpts don't contain the answer say exactly that; if a value is `[***]` say it is redacted; if a date is `[·]` say it is a placeholder; you are not giving legal advice.
3. Few-shot: one good answer example, one correct-refusal example.
4. Build the tricky-question set into `eval/questions.jsonl` (~15 questions): normal ones, the redacted-price trap, the placeholder-date trap, a not-in-any-document question, a cross-document comparison.
5. Loop: run set → inspect with `debug=true` → adjust prompt → rerun. Log which prompt version produced which output.

**Done when:** all trap questions get correct refusals/flags with the current prompt version, and normal questions keep section-level citations.

---

## 6. Hallucination prevention (layered guards)

**Goal:** mechanisms *around* the model, not just instructions *to* it. Each layer is code you can point at in the interview.

1. **Retrieval gating** (`guards.py`): if the top fused score is below a threshold, short-circuit to "no relevant clause found" — the LLM never gets a chance to improvise. Tune the threshold with the not-in-corpus eval question.
2. **Grounded prompting** — phase 5, already done.
3. **Citation verification** (post-generation): every citation in the structured output must reference a chunk that was actually retrieved; optionally check answer/chunk word overlap. Failures → flagged or regenerated. Cheap, deterministic, no extra LLM calls.
4. **Faithfulness judge** (`eval/run_eval.py`): LLM-as-judge with `JUDGE_MODEL` ≠ answer model (reduces self-preference bias): split the answer into claims, ask the judge whether each claim is supported by the retrieved excerpts → faithfulness score per answer, table across the eval set. This is the "methodical approach" bonus; cite RAGAS as the reference method you reimplemented minimally.
5. **(Optional) UQLM consistency scoring in the eval harness only** — see decision log below.

**Done when:** eval report table exists (question, hit@5, faithfulness, trap pass/fail) and is pasted into the assessment report.

### 6.5 PII masking (Presidio)

`uv add presidio-analyzer presidio-anonymizer` (+ spaCy model). Mask retrieved chunk text before it leaves for the gateway (`PERSON_1`, `ORG_2`, reversible mapping), unmask after. Keep it a separate step with its own debug script (`scripts/mask.py "text"`) and an env kill-switch `MASKING_ENABLED` — masking bugs are confusing, so you want to A/B it off instantly.

---

## 7. Dockerising and testing

**Goal:** `docker compose up` → working system; test suite green.

1. Dockerfile: multi-stage from `ghcr.io/astral-sh/uv` image; `uv sync --frozen --no-dev`; copy only `app/` and `scripts/`. Pre-download the fastembed model into the image so first startup isn't slow/network-bound.
2. `compose.yml`: `app` + `qdrant` services, named volume for Qdrant storage, healthcheck on Qdrant, `depends_on: condition: service_healthy`, key passed via `env_file: .env`. An `ingest` one-shot service (or startup hook) to index the PDFs.
3. Tests: unit (extraction cleaning, chunk boundaries, RRF ordering, citation verifier), integration (against live Qdrant), eval run as the acceptance test.
4. README: two paths — `docker compose up` and local `uv run`; `.env.example`; example curl.

**Done when:** fresh clone + `.env` + `docker compose up` answers a question; the submission zip contains no `.venv`, no `.env`, no model weights, no `qdrant_data/`, no `data/` outputs — and not the assessment brief PDF or this blueprint.

### 8 (later). UI
Thin static page or Streamlit hitting `POST /ask`. Zero pipeline changes — the structured JSON response was designed for this.

---

## Decision log: UQLM and LangChain

**UQLM** ([cvs-health/uqlm](https://github.com/cvs-health/uqlm)) detects hallucinations via *uncertainty quantification* — mainly by sampling several answers to the same question and measuring how consistent they are (an inconsistent model is an unsure model), plus token-probability ("white-box") and LLM-judge scorers.

Pros: research-backed and citable (JMLR paper) — strong "methodical" credibility; black-box scorers work with any gateway model; gives a confidence number you could return in the API.

Cons: consistency scorers need 5–10 generations per question — 5–10× cost and latency, unusable in the live `/ask` path; it measures *self-consistency*, not *groundedness to your retrieved clauses* — a model can consistently repeat the same hallucination, and faithfulness-to-source is what the assessment actually asks for; white-box scorers need token log-probabilities, which the gateway may not expose for every model; and it pulls the LangChain ecosystem in as a dependency.

**Verdict:** don't put it in the request path. Optionally add as a *secondary* metric in the offline eval harness (`uv add --group eval uqlm`) so the report can say "citation verification + faithfulness judge, cross-checked with UQLM semantic-consistency scores." Nice-to-have, phase 6 step 5, only if time remains.

**LangChain** for prompting/orchestration.

Pros: fast scaffolding (prompt templates, output parsers, Qdrant integration), easy model swapping, signals ecosystem familiarity.

Cons: heavy dependency graph with a history of breaking changes; stack traces run through abstraction layers — directly against the "small steps, debuggable" goal of this blueprint; the pipeline is ~6 focused modules, and hiding them inside chains hides exactly the understanding the interview wants to see; prompt templating at this scale is f-strings/jinja; custom middle steps (masking, citation verification, gating) fit awkwardly into chain abstractions; the scope doc already rejected RAGAS for being "less explainable in an interview" — the same logic applies harder here.

**Verdict:** skip it for the core pipeline; plain `openai` client + prompt files. Mention in the report that you deliberately avoided frameworks to keep every stage inspectable — that's a reasoning point in your favor, and interviewers reliably respect it. (Irony worth knowing: adopting UQLM brings `langchain-core` in anyway — another reason to confine UQLM to the optional eval group.)

---

## Suggested build order & effort

| Phase | Effort | Risk if skipped/rushed |
|---|---|---|
| 0 Scaffold | 0.5 h | debugging chaos later |
| 1 Extraction | 2 h | garbage in → everything downstream lies |
| 2 Chunking | 3 h | citations impossible — required feature fails |
| 3 Retrieval + mini-eval | 3 h | can't tell retrieval bugs from prompt bugs |
| 4 LLM integration | 2 h | — |
| 5 Prompts + golden set | 3 h | trap questions fail silently |
| 6 Guards + judge eval | 3–4 h | bonus points lost |
| 6.5 Masking | 2 h | differentiator lost (cut first if time-boxed) |
| 7 Docker + tests + README | 3 h | "does it run" is the first filter |

Rule of thumb: never start phase N+1 while phase N's debug script output still looks wrong — with five documents, every stage is small enough to verify by eye.
