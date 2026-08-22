# Contract RAG

Ask questions about a small corpus of commercial contracts and get answers
cited down to the specific section — with a refusal instead of a guess when
the contracts do not say.

Five agreements are indexed: a joint venture, a manufacturing agreement, a
hosting agreement, a trademark licence and a transportation agreement.

---

## Run it with Docker

```bash
cp .env.example .env      # copy .env.example .env  on Windows
# paste your gateway key into AI_GATEWAY_API_KEY
docker compose up --build
```

Three services start in order: Qdrant comes up, a one-shot `ingest` service
reads the PDFs and builds the index, then the API starts. First build takes a
few minutes — it downloads dependencies and bakes the embedding models into
the image. After that, startup is seconds.

Open **http://localhost:8000** for the interface, or
**http://localhost:8000/docs** for the API.

```bash
docker compose down       # stop, keep the index
docker compose down -v    # stop, discard the index
```

## Run it locally

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env      # and paste your key in

docker compose up -d qdrant     # or set QDRANT_PATH in .env to run without Docker

uv run scripts/extract.py       # PDFs  -> text
uv run scripts/chunk.py         # text  -> clause-aware chunks
uv run scripts/index.py         # chunks -> Qdrant

uv run uvicorn app.api:app --reload
```

The first `index.py` downloads the embedding model (~130MB) and takes a minute
or two. After that it is seconds.

Without Docker at all: set `QDRANT_PATH=./qdrant_local` in `.env` and Qdrant
runs embedded inside the process. One process at a time, so stop the API
before re-indexing.

## Ask it something

```bash
curl -s localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "What are the confidentiality obligations in the hosting agreement?"}'
```

```json
{
  "answer": "Under the Hosting Agreement, each party must hold the other's Confidential Information in strict confidence ... (Section 9.1)",
  "citations": [{"doc_title": "Hosting Agreement", "section": "Section 9.1", "page": 12}],
  "answered": true,
  "excerpts_used": 8
}
```

Useful additions:

| | |
|---|---|
| `?debug=true` | also returns the retrieved chunks and the exact prompt that was sent |
| `"doc_id": "hosting"` | restrict the search to one contract; a fragment is enough |
| `"history": [...]` | previous turns, for follow-up questions |
| `POST /ask/stream` | the same answer, with progress reported stage by stage |
| `GET /health` | active configuration, contracts found, chunks indexed |
| `GET /documents` | what is indexed, with the identifiers `doc_id` accepts |

Nothing is stored between requests. A conversation is carried by the client
posting its own history back, so there is no session state and no database.

## Configuration

Every setting is read from the environment. The code holds no fallback values:
a missing variable stops the application at startup with a message naming it,
rather than quietly running on a setting nobody chose.

`.env.example` is the complete template, with a note on each variable
explaining what it does and, where a value was measured rather than guessed,
where the measurement is written up. The ones worth knowing about:

| Variable | Why it is set where it is |
|---|---|
| `SEARCH_MODE=hybrid` | Dense retrieval alone misses the redacted-price and placeholder-date clauses entirely. See `docs/retrieval-eval.md`. |
| `TOP_K=8` | At 5, the trademark licence's Section 4.4 fell one slot outside the window and the answer silently omitted a termination trigger. |
| `MIN_SCORE=0.56` | Calibrated, not guessed: `uv run scripts/calibrate_gate.py`. It rejects questions that are not about these contracts. It cannot tell whether the corpus can answer a contract question — that limit is deliberate and documented. |
| `PROMPT_VERSION=v5` | Five versions, each measured against the same question set. `docs/answer-eval.md` has the progression. |
| `MASKING_ENABLED=false` | Personal-data masking is off unless asked for. See below. |

`.env` holds the gateway key and is never committed.

## Personal-data masking

Names and contact details can be replaced with stable placeholders before
contract text is sent to the model, and restored in the answer afterwards.
It ships switched off.

It needs a spaCy language model that is too large to put in the image by
default. To include it:

```bash
INSTALL_MASKING=true docker compose build
```

Locally: `uv run python -m spacy download en_core_web_lg`. Then set
`MASKING_ENABLED=true`.

What is *not* masked matters more than what is. Dates, locations and
organisations are left alone: masking them would remove the answer to "when
does this take effect" and "which state's law governs". `app/safety/masking.py`
explains the reasoning with the counts that produced it.

## Evaluation

```bash
uv run eval/retrieval_eval.py --md          # does the right clause come back
uv run eval/answer_eval.py --runs 3 --md    # is the answer right, across prompt versions
uv run eval/faithfulness_eval.py --runs 3   # is every claim supported by a cited excerpt
uv run pytest                               # 196 tests
```

Measured on the current build:

| | |
|---|---|
| Retrieval, 21 questions, top-8 | hybrid **100%** hit rate, MRR 0.89 — against 90% dense, 95% sparse |
| Answers, 26 questions × 3 runs | 24/26 correct, 0 unsupported citations reached the reader |
| Faithfulness, 87 claims | 96.4% supported by a cited excerpt |
| Judge self-test | caught 4/4 planted fabrications, left the clean control alone |

`docs/results.md` collects these; `docs/how-it-works.md` walks the pipeline.

## How it works

```
PDF ──► text + page numbers ──► clause-aware chunks ──► Qdrant
                                                          │
question ──► condense (if it is a follow-up) ──► retrieve ─┘
                                    │
                       relevance gate: too far off-topic, refuse here
                                    │
                          mask ──► prompt ──► model
                                    │
                    verify every citation against what was retrieved
                                    │
                                 answer
```

Two things are worth calling out because they were not obvious in advance.

**Every chunk is stored twice**, as a dense vector and as a sparse one, and
searches are fused by reciprocal rank. The first evaluation said this was
pointless — dense alone scored higher. Adding questions about clauses where
the value is redacted or left as a placeholder reversed it: a dense vector
averages a whole chunk, so one fact inside a long heterogeneous chunk gets
diluted, while term matching does not average.

**Citations are verified, not trusted.** Every section the model names is
matched against what it was actually shown. One that does not match is
dropped before the reader sees it, and the model gets a single chance to
correct itself first.

No orchestration framework. The pipeline is six steps that a person can read
end to end, and each one is a place a wrong answer can be traced to.

## If something goes wrong

**"Configuration is incomplete"** — a variable in `.env.example` is missing
from your `.env`. The message names it. Every setting is required; nothing
falls back to a value in the code.

**503, "No chunks are indexed"** — the ingest step has not run, or ran against
a different Qdrant than the one the API is talking to. Check `GET /health`:
it reports the collection and how many chunks are in it.

**The container cannot load the embedding model** — the image is built with
`HF_HUB_OFFLINE=1`, so a cache miss fails loudly instead of quietly
downloading 130MB on the first question. It means `DENSE_MODEL` or
`SPARSE_MODEL` in `.env` no longer matches what was baked in. Rebuild:
`docker compose build --no-cache`.

**Changed `CHUNK_MAX_WORDS` or either model** — the stored index no longer
agrees with the configuration. Re-run `extract.py`, `chunk.py`, `index.py`,
or `docker compose down -v && docker compose up`.

## Layout

```
app/
  api.py            HTTP endpoints, and the static interface
  answer.py         the pipeline: condense, retrieve, gate, prompt, verify
  config.py         every setting, in one place
  ingest/           PDF extraction, clause-aware chunking
  search/           embeddings, the Qdrant collection, hybrid retrieval
  generate/         the model client, prompt rendering, prompt files by version
  safety/           citation and refusal guards, personal-data masking
scripts/            one debug CLI per stage - run any step on its own
eval/               the measurement harnesses and the question set
tests/              196 tests, no network, no Docker
docs/               results, and the write-up behind each number
```
