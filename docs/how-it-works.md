# How this system works: the libraries, the jargon, and the numbers

Written to be read start to finish. Part 1 is every library and why it is here.
Part 2 explains the words that get thrown around in retrieval work. Part 3
explains every score this project produces, including what each one cannot tell
you — which is usually the more useful half.

---

## Part 1 — The libraries

### uv — environment and dependency manager

Replaces `pip`, `venv` and `pip-tools` with one tool. Two things it gives us
that matter here.

`uv.lock` records the exact version of every package, including packages your
packages depend on. Anyone who clones this repo and runs `uv sync` gets a
byte-identical environment. Without a lockfile, "it works on my machine" is a
coin toss — you install `fastapi` today and I install it next month, and we are
running different code.

`uv run <command>` executes inside the project's environment without you having
to activate anything. That is why every command in this project starts with
`uv run`.

### FastAPI + uvicorn — the web layer

**FastAPI** turns Python functions into HTTP endpoints. You annotate a function
with types, and it validates incoming requests against them for free: a request
missing `question`, or sending a number where a string belongs, is rejected
before your code runs, with a message saying exactly what was wrong.

It also generates the interactive documentation at `/docs` from those same type
annotations. That page is not something we wrote — it is derived from the code,
so it cannot go out of date.

**uvicorn** is the server that actually runs FastAPI. FastAPI describes what the
endpoints do; uvicorn accepts the network connections and speaks HTTP. You need
both, the way you need both an engine and a car.

### pydantic and pydantic-settings — typed data

**pydantic** is what FastAPI uses underneath for validation. We use it directly
for the shape of a model's answer (`ModelAnswer` in `app/answer.py`): the model
returns text that should be JSON, and pydantic checks it really has an `answer`
string and a list of citations before any other code touches it.

**pydantic-settings** is the same idea applied to configuration. `app/config.py`
declares every setting with a type and a default; the values come from
environment variables or the `.env` file. The point is type conversion:
environment variables are *always* strings, so `os.getenv("TOP_K")` gives you
`"8"`, not `8`. Worse, `os.getenv("MASKING_ENABLED")` gives you the string
`"false"`, and a non-empty string is *true* in Python — so a naive kill switch
never switches anything off. pydantic converts properly and fails loudly at
startup if a value is malformed.

### PyMuPDF — reading the PDFs

Extracts text from PDF files, page by page. Fast, and unlike some alternatives
it preserves the reading order well.

A quirk worth knowing: it used to be imported as `fitz` (the original name of
the underlying library), and most tutorials online still say `import fitz`. That
name is deprecated; this project uses `import pymupdf`.

What it does *not* do is understand documents. It gives you the characters on
the page. Everything about sections, headings and page furniture is our own code
in `app/extraction.py` and `app/chunking.py`.

### Qdrant + qdrant-client — the vector database

A **vector database** stores lists of numbers and answers the question "which of
these stored lists is most similar to this one?" quickly. That is the entire job.
Ordinary databases can find exact matches; a vector database finds *close*
matches in meaning.

Qdrant specifically, for three reasons:

- It stores **two vectors per record under different names**, which is what lets
  one search cover both meaning and exact wording (see hybrid search below).
- It performs the **fusion of two result lists server-side**, so one network
  round trip does the whole hybrid search rather than us merging lists in Python.
- The Python client can run Qdrant **embedded**, inside your own process,
  storing to a local folder. That is why this project works without Docker while
  you build it, and switches to a real server by changing one environment
  variable.

### fastembed — turning text into vectors, locally

Runs the embedding models on your own CPU. No API calls, no data leaving the
machine to be embedded, no per-token cost. It uses ONNX Runtime underneath — a
runtime for running trained models efficiently outside the framework they were
trained in.

Two models, doing different jobs:

- **BAAI/bge-small-en-v1.5** produces the dense vectors (meaning). 384 numbers
  per chunk, roughly 130MB of model, a few milliseconds per chunk.
- **Qdrant/bm25** produces the sparse vectors (exact wording).

### openai — the client, pointed somewhere else

The `openai` package is just an HTTP client that speaks a particular request
format. Because the Vercel AI Gateway accepts that same format, we point the
client at the gateway's address and can then use models from any provider —
OpenAI, Anthropic, Alibaba — by changing a string in `.env`. No vendor-specific
code anywhere in the project.

That is why `LLM_MODEL` and `JUDGE_MODEL` can be different models from different
companies without a second library.

### pytest and ruff — development only

**pytest** runs the test suite. Tests here are not decoration: several of them
pin behaviour that was broken and fixed (heading detection, citation matching),
so if someone "simplifies" that code later, the test says what broke and why.

**ruff** is a linter and formatter — it checks style and catches dead code.

Both are in a dependency group called `dev`, which means `uv sync` installs them
while the Docker build with `--no-dev` skips them. The production image does not
need a test runner.

### hatchling — the build backend

The thing that packages `app/` into an installable unit. You never call it
directly. It exists so `import app.config` works from anywhere in the project
rather than only from the top folder.

### Deliberately not used

**LangChain** would have scaffolded this faster. It was skipped because every
stage here is small enough to read, and hiding those stages inside a framework's
abstractions removes exactly the understanding this project is meant to
demonstrate. Prompt templating at this scale is string replacement. Stack traces
that run through six layers of framework are worse than stack traces that don't.

**RAGAS** is the standard library for evaluating retrieval systems. We
reimplemented one metric from it — faithfulness — in about eighty lines, because
being able to explain exactly how a number was produced is worth more here than
importing it.

---

## Part 2 — The jargon

### Embedding, vector, dimension

An **embedding** is a list of numbers representing a piece of text, produced by
a model trained so that texts with similar meanings get similar lists. Each
number is a **dimension**; our model produces 384 of them.

There is no way to look at the numbers and say what any single one means. What
matters is only distance: two texts about termination end up close together in
that 384-dimensional space, and a text about gas quality ends up far away.

### Cosine similarity

The measure of "close together". It compares the *direction* of two vectors,
ignoring their length, and gives a number between -1 and 1 — in practice for
this kind of text, between about 0.3 and 0.9.

Length is ignored on purpose: a two-sentence clause and a two-page clause about
the same subject point the same way even though one has much more text in it.

### Chunk

A contract is too long to embed as one vector — averaging a twenty-page document
into 384 numbers loses everything specific. So documents are split. **How** you
split is the design decision: this project splits at section boundaries, so
every chunk corresponds to a numbered clause and can carry the section number
that a citation needs. Splitting every 500 characters would be easier and would
make section-level citation impossible.

### Dense vs sparse vectors

**Dense** vectors are the 384 numbers above: mostly non-zero, capturing meaning.
They find a termination clause when you ask "can we end this early?", even
though neither word appears in it.

**Sparse** vectors have one position per word in the vocabulary, almost all
zero. They capture *which exact words* appear. They find the clause containing
the literal word "Transporter".

Neither is better. They fail differently, which is the argument for using both.

### BM25 and IDF

**BM25** is the classic keyword scoring formula behind most search engines. Two
ideas in it: a document matching your search word more often scores higher, and
a *rare* word counts for more than a common one.

That second idea is **IDF** — inverse document frequency. If a word appears in
every document, knowing it appears in one particular document tells you nothing,
so its weight drops towards zero. This matters concretely here: "Transporter"
appears in nearly every chunk of the transportation agreement, so its IDF
collapses and BM25 becomes bad at finding the one clause that defines it. The
textbook argument for keyword search over legal text — that defined terms are
rare words — turns out to be backwards, because a contract defines a term
precisely so it can use it constantly.

### Hybrid search and RRF

**Hybrid search** means running both searches and combining the results.

The combination is the hard part, because a cosine similarity of 0.78 and a BM25
score of 14.2 are not on the same scale, and BM25's range shifts with the corpus.
Adding them requires inventing a weighting you then have to defend.

**Reciprocal Rank Fusion (RRF)** sidesteps this by throwing the scores away and
using only each result's *position* in each list. A chunk ranked first
contributes 1/(60+1), one ranked fifth contributes 1/(60+5), and the two
contributions are added. Anything both searches like rises to the top. Nothing
needs normalising, and there is no weight to justify. The 60 is a constant that
stops the top one or two positions from dominating everything else.

### top_k

How many chunks get retrieved and put into the prompt. Ours is 8.

Too few and the answer is confidently incomplete — at 5, the trademark licence's
Section 4.4 fell one place outside the window and the answer silently omitted a
way the contract can terminate. Too many and irrelevant text dilutes the prompt
and invites the model to drift.

### Temperature

How much randomness the model uses when choosing each next word. 0 means "always
take the most likely option". We use 0 because a system whose answer changes
between runs cannot be evaluated meaningfully.

Note carefully: temperature 0 reduces variation, it does not eliminate it. We
measured the same question producing different answers on different runs at
temperature 0, which is why the evaluation supports `--runs 3`.

### Grounding, and hallucination

An answer is **grounded** if everything it asserts comes from the source text
provided. A **hallucination** here is not "the model said something false about
the world" — it is "the model said something that is not in the excerpts". A
statement that is perfectly true of contracts in general but absent from these
five is a hallucination for our purposes, because the user cannot check it
against a section.

---

## Part 3 — Every score, and what it does not tell you

### hit@k — retrieval

Of the questions where we know which section holds the answer, the share where a
correct section appeared in the top *k* results.

**hit@1** is "the very first result was right". **hit@5** is "a right answer was
somewhere in the first five". hit@5 matters most for us, because we send the
model 8 excerpts — the question is whether the right clause was in the window at
all, not whether it was first.

Our measured numbers, 19 scored questions:

| mode | hit@1 | hit@3 | hit@5 |
|---|---|---|---|
| hybrid | 79% | 95% | **100%** |
| dense only | 84% | 89% | 89% |
| sparse only | 74% | 84% | 89% |

**What it does not tell you:** whether the answer built from those chunks was any
good. hit@5 of 100% only means the raw material reached the model.

### MRR — mean reciprocal rank

Take the position of the first correct result, invert it (rank 1 gives 1.0,
rank 2 gives 0.5, rank 4 gives 0.25), average across questions. It rewards
putting the right clause *first*, not merely somewhere.

Useful when two systems have the same hit@5 but one consistently ranks better.

**What it does not tell you:** anything about the questions it missed entirely —
they contribute 0 and vanish into the average.

### Cosine similarity, as the relevance gate

A number from roughly 0.3 to 0.9 measuring how close the question is to the
nearest chunk. We refuse before calling the model when it falls below
`MIN_SCORE` (0.56 here).

Measured on this corpus: genuine questions score 0.63 to 0.89; questions about
baking, weather and Python score 0.47 to 0.55.

**What it does not tell you — and this is the important one:** whether the corpus
can actually answer the question. "What is the notice period for terminating the
office lease?" scored **0.668**, higher than two genuine questions, because it is
contract-shaped language whether or not a lease exists in the corpus. Similarity
measures topic, not answerability. The gate filters queries that are not about
contracts; it cannot filter contract questions this corpus cannot answer. That
job belongs to the prompt rules and citation checking.

### The fused RRF score

The number in the `score` column when hybrid search is on. It is **not** a
similarity. It is a rank-agreement score, and something always has to come
first — so the top result of a nonsense query scores about as well as the top
result of a good one. Use it to compare results *within one query*, never as a
confidence measure across queries. That is exactly why the gate uses cosine
similarity instead.

### Answer evaluation — pass rate and traps

22 questions, each run 3 times, majority vote. Deterministic checks only: did it
cite the expected section, did it use the right language for a redaction.

| prompt | passed | traps | rejected citations | unstable |
|---|---|---|---|---|
| v1 | 17/22 | 1/5 | 1 | 0 |
| v2 | 20/22 | 3/5 | 0 | 1 |
| v3 | 21/22 | 4/5 | 0 | 0 |

**Traps** are the five questions where the honest answer is "the contracts do not
give you this", in four disguises: the value is redacted `[***]`, the date is a
blank placeholder `[·]`, the topic is absent entirely, or the clause exists but
defers the substance to a document we do not have. These are worth more than the
straightforward questions, because getting them wrong produces a confident,
plausible, wrong answer rather than an obviously bad one.

**Rejected citations** counts citations the model produced that named a section
it was never shown. Those are removed before the answer is returned.

**Unstable** counts questions that passed some runs and failed others — the
honest measure of how much of a score is luck.

**What it does not tell you:** whether the substance of an answer is supported.
These checks look at citations and keywords; an answer can cite the right section
and still say something the section does not support.

### Faithfulness — the judge

For that last gap: a second model reads each answer and the excerpts it was built
from, breaks the answer into individual claims, and marks each claim supported or
unsupported. **Faithfulness = supported claims ÷ total claims.**

The judge is a *different* model from the answering model (`JUDGE_MODEL`), because
a model asked to grade its own output tends to agree with itself.

**What it does not tell you:** whether the answer is true in the world, or
whether it is complete. An answer that faithfully reports one clause while
missing a second relevant one scores 100%. And the judge is itself a language
model — two models trained on overlapping data can share a blind spot. It is a
check, not a proof.

---

## The shape of the whole thing

```
PDF files
  |  PyMuPDF                     extract text, keep page numbers
  v
cleaned text per page
  |  our own regexes             find section boundaries per document
  v
120 chunks, each carrying document, section, heading, pages
  |  fastembed (two models)      dense vector + sparse vector each
  v
Qdrant collection
  |  hybrid query + RRF          8 chunks most likely to hold the answer
  v
relevance gate                   below 0.56 similarity: refuse, no model call
  |
  v
prompt v3 + excerpts
  |  openai client -> gateway    temperature 0
  v
JSON answer with citations
  |  citation verification       drop any citation naming an unseen section
  v
answer the user sees
```

Four places a wrong answer can come from, and each has its own way to look:
`scripts/extract.py`, `scripts/chunk.py`, `scripts/search.py`, and
`scripts/ask.py --debug`. That separation is the reason a bad answer is a
ten-minute investigation rather than a guess.
