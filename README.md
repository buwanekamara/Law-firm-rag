# Contract RAG

Ask questions about a folder of contracts and get answers that cite the exact
section they came from. When the contracts do not say, it says so instead of
guessing.

Five agreements are indexed out of the box: a joint venture, a manufacturing
agreement, a hosting agreement, a trademark licence and a gas transportation
agreement.

```
You:  What are the confidentiality obligations in the hosting agreement?
It:   Each party must hold the other's Confidential Information in strict
      confidence and use it only to perform the agreement.
      - Hosting Agreement, Section 9.1 (page 12)
```

Every citation is checked against the text the model was actually shown. One
that does not match is removed before you see it.

---

## Contents

- [Adding your own contracts](#adding-your-own-contracts)
- [Run it with Docker](#run-it-with-docker)
- [Install with uv](#install-with-uv)
- [Install with pip](#install-with-pip)
- [Using it](#using-it)
- [Settings](#settings)
- [What each file does](#what-each-file-does)

---

## Adding your own contracts

1. Drop the PDFs into the `contracts/` folder. Nothing else needs editing.
2. Rebuild the index:

   ```bash
   uv run scripts/extract.py    # PDFs   -> cleaned text
   uv run scripts/chunk.py      # text   -> one chunk per contract section
   uv run scripts/index.py      # chunks -> searchable index
   ```

   With Docker, `docker compose up --build` does all three on startup.

3. Check it worked: `uv run scripts/chunk.py` prints a table of every chunk.
   The section column should read like the contract's own table of contents.

**Name the files carefully.** The title shown in citations comes from the last
part of the filename, after the final `_` or `-`, so
`Acme_2019_EX-10.1_Supply Agreement.pdf` is cited as "Supply Agreement".
A file called `scan1.pdf` will be cited as "scan1".

**Scanned PDFs will not work.** Text is read from the PDF's own text layer.
A scanned image of a contract produces nothing to index.

One thing to change if you swap the corpus: the refusal message in
`app/answer.py` names the five agreements out loud, so a question outside the
corpus gets a useful reply rather than "nothing found". Edit `_SCOPE` there to
describe your own set.

---

## Run it with Docker

The shortest path. Needs Docker and an AI Gateway key.

```bash
cp .env.example .env      # Windows: copy .env.example .env
# open .env and paste your key into AI_GATEWAY_API_KEY

docker compose up --build
```

Then open **http://localhost:8000**.

The first build takes a few minutes: it installs dependencies and bakes the
embedding models into the image. After that, startup is seconds. Three things
start in order — the search engine, a one-shot job that reads the contracts
and builds the index, then the web service.

```bash
docker compose down       # stop, keep the index
docker compose down -v    # stop, throw the index away
docker compose logs -f    # watch what it is doing
```

---

## Install with uv

[uv](https://docs.astral.sh/uv/) installs the exact versions in `uv.lock`.
Needs Python 3.12.

```bash
uv sync
cp .env.example .env      # then paste your key in

docker compose up -d qdrant      # the search engine
uv run scripts/extract.py
uv run scripts/chunk.py
uv run scripts/index.py

uv run uvicorn app.api:app --reload
```

**No Docker at all?** Set `QDRANT_PATH=./qdrant_local` in `.env` and the search
engine runs inside the application, storing to that folder. Only one process
may use it at a time, so stop the web service before re-indexing.

The first `index.py` downloads the embedding model (~130MB). After that it
takes seconds.

---

## Install with pip

Same thing without uv. Needs Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS, Linux:  source .venv/bin/activate

pip install -e .
cp .env.example .env               # then paste your key in

docker compose up -d qdrant        # or set QDRANT_PATH as above
python scripts/extract.py
python scripts/chunk.py
python scripts/index.py

python -m uvicorn app.api:app --reload
```

For the tests and the linter as well:

```bash
pip install pytest pytest-asyncio ruff
pytest
ruff check .
```

pip resolves versions itself rather than reading `uv.lock`, so you may get
newer releases than the ones this was built against. If something behaves
oddly, that is the first thing to check.

---

## Using it

**The web page** at http://localhost:8000 — ask a question, pick one contract
or search all of them, tick "Show prompt and sources" to see which sections
were retrieved and the exact prompt that was sent.

**The API** at http://localhost:8000/docs, or directly:

```bash
curl -s localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "What are the confidentiality obligations?"}'
```

| | |
|---|---|
| `POST /ask` | ask a question |
| `POST /ask/stream` | the same, reporting progress while it works |
| `GET /health` | what configuration is in effect, how many chunks are indexed |
| `GET /documents` | which contracts are indexed |

Useful fields in the request body: `"doc_id": "hosting"` restricts the search
to one contract, `"top_k": 12` changes how many excerpts are used, and
`"history": [...]` carries previous turns so follow-up questions work. Add
`?debug=true` to the URL to get the retrieved chunks and the exact prompt back
alongside the answer.

Nothing is stored between requests. A conversation is carried by the client
sending its own history back, so there is no database and no session state.

**From the command line**, without starting the service:

```bash
uv run scripts/ask.py "When can the licence be terminated?"
uv run scripts/ask.py "What is the price?" --doc manufacturing --debug
uv run scripts/search.py "force majeure" --k 10     # retrieval only, no model
```

---

## Settings

Everything is configured through `.env`. There are no fallback values in the
code, so a missing variable stops startup with a message naming it.

`.env.example` is the full list. Each entry says what it accepts and what
changes when you change it. The ones people usually touch:

| Variable | Options | What it does |
|---|---|---|
| `LLM_MODEL` | any gateway model id | which model writes the answers |
| `TOP_K` | 1–20 | how many contract sections the model is shown |
| `SEARCH_MODE` | `dense` `sparse` `hybrid` | search by meaning, by exact words, or both |
| `MIN_SCORE` | 0.0–1.0 | how far off-topic a question must be to be refused; 0 turns it off |
| `PROMPT_VERSION` | `v1`–`v5` | which set of answering rules to use |
| `MASKING_ENABLED` | `true` `false` | replace names and contact details before sending text out |

Changing `CHUNK_MAX_WORDS`, `DENSE_MODEL` or `SPARSE_MODEL` means the stored
index no longer matches the settings — re-run the three ingest scripts.

To see the effect of a change instead of guessing:

```bash
uv run eval/retrieval_eval.py            # does the right section come back
uv run eval/answer_eval.py               # is the answer right
uv run scripts/calibrate_gate.py         # pick MIN_SCORE from your own data
uv run scripts/chunk.py --sizes          # what a chunking change did
```

`docs/` holds the results of those runs and the reasoning behind each setting.

---

## What each file does

```
contracts/              the source PDFs - put yours here
data/                   generated: extracted text and chunk files
docs/                   evaluation results and design notes

app/
  api.py                the web service: /ask, /health, /documents
  answer.py             the pipeline, end to end
  config.py             every setting, read from the environment
  static/index.html     the web page - one file, no build step

  ingest/
    extraction.py       PDF -> clean text, keeping page numbers
    chunking.py         text -> one chunk per contract section

  search/
    embeddings.py       turns text into vectors, locally
    indexing.py         builds and loads the search index
    retrieval.py        finds the sections most likely to answer

  generate/
    llm.py              the model call
    prompting.py        loads prompt files and fills in the question
    conversation.py     rewrites follow-up questions so they can be searched
    prompts/            the answering rules, one file per version

  safety/
    guards.py           checks citations are real and refusals are genuine
    masking.py          replaces names and contact details before sending

scripts/                one debug tool per stage - run any step on its own
  extract.py  chunk.py  index.py  search.py  ask.py  mask.py  calibrate_gate.py

eval/                   measurement harnesses and the question set
  retrieval_eval.py     does the right section come back
  answer_eval.py        is the answer right, across prompt versions
  faithfulness_eval.py  is every claim backed by a cited excerpt
  questions.jsonl       the questions and their expected answers

tests/                  202 tests, no network and no Docker needed
```

**Where to start reading:** `app/answer.py` is the whole pipeline in one file,
and every other module is a step in it.

---

## If something goes wrong

**"Configuration is incomplete"** — a variable from `.env.example` is missing
from your `.env`. The message names it.

**503, "No chunks are indexed"** — the ingest scripts have not been run, or
were run against a different search engine than the one the service is using.
`GET /health` shows which one it is talking to and how many chunks it sees.

**"AI_GATEWAY_API_KEY is not set"** — the key is missing from `.env`.
Everything except the final answer works without it, so search and the ingest
scripts will still run.

**Answers cite the wrong sections** — check retrieval first:
`uv run scripts/search.py "your question"`. If the right section is not in
that list, no prompt change will fix it.
