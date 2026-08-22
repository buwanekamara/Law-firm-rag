"""Single source of truth for configuration.

Every tunable value in the project lives here: model names, retrieval knobs,
feature switches. None of them carry a fallback value in code. They are read
from the environment (or from a local .env file), and a missing one is a
startup error rather than a silent default, so what the application is running
with is always exactly what the environment says.

The two folder paths are the deliberate exception: they are derived from the
location of this file, so the project works the same whether it is checked out
to a laptop or copied into a container image.

Nothing secret is hard-coded - the gateway key only ever arrives via the
environment.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# This file is app/config.py, so one level up is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- paths -------------------------------------------------------------
    # Derived, not configured: see the module docstring.
    contracts_dir: Path = PROJECT_ROOT / "contracts"
    data_dir: Path = PROJECT_ROOT / "data"

    # --- LLM, via the Vercel AI Gateway ------------------------------------
    ai_gateway_api_key: str
    ai_gateway_base_url: str
    llm_model: str
    # A different model judges the answers in the offline evaluation, so the
    # answering model is not grading its own homework.
    judge_model: str
    llm_temperature: float

    # --- vector store ------------------------------------------------------
    qdrant_url: str
    # Set this instead of running a server and Qdrant runs embedded, inside
    # this process, storing to the given folder. See app.search.indexing.get_client.
    qdrant_path: str
    qdrant_collection: str
    dense_model: str
    sparse_model: str

    # --- retrieval ---------------------------------------------------------
    # "dense", "sparse" or "hybrid". Hybrid is the measured choice: dense
    # alone misses the redacted-price and placeholder-date clauses entirely.
    # See the docstring at the top of app/search/retrieval.py and
    # docs/retrieval-eval.md.
    search_mode: str
    # 8, not 5: measured. At 5 the trademark licence's Section 4.4 (an
    # automatic termination trigger) fell one slot outside the window and the
    # answer silently omitted it. See docs/retrieval-eval.md.
    top_k: int
    # Relevance gate. If the closest chunk's DENSE cosine similarity is below
    # this, the question is refused before the model is called at all. Not the
    # fused hybrid score: RRF ranks by agreement between two result lists, so
    # the top hit of a nonsense query scores about as well as the top hit of a
    # good one. 0 disables the gate - run scripts/calibrate_gate.py to pick a
    # value from data rather than guessing one.
    min_score: float
    # Give the model one chance to correct a citation that names a section it
    # was never shown, before the citation is dropped.
    citation_retry: bool
    # Each half of a hybrid search fetches this multiple of top_k before the
    # two rankings are fused. Fusing two top-5 lists gives the ranking little
    # to work with; fusing two top-32 lists lets a chunk that placed 12th on
    # one side and 2nd on the other still surface.
    prefetch_multiplier: int

    # --- ingestion ---------------------------------------------------------
    # Chunk size in words. bge-small accepts 512 tokens and English runs about
    # 1.3 tokens per word, so 350 leaves headroom rather than silently
    # truncating the tail of a clause at embedding time.
    #
    # Changing either of these makes the stored chunks disagree with the
    # configuration. Re-run: scripts/extract.py, scripts/chunk.py, scripts/index.py
    chunk_max_words: int
    chunk_overlap_words: int
    # Embedding batch size. Lower it on a memory-constrained host.
    embed_batch_size: int

    # --- conversation ------------------------------------------------------
    # How many previous exchanges are carried into a follow-up. Older context
    # stops helping and starts dragging retrieval towards whatever was
    # discussed earliest.
    history_turns: int

    # --- gateway -----------------------------------------------------------
    # Generous on purpose: a slow gateway is a worse failure than a slow answer.
    llm_timeout_seconds: float

    # --- prompts and guards ------------------------------------------------
    # v5 is the measured choice. v1 mishandled every "the value is not here"
    # case; v2 added the redaction and placeholder rules; v3 added the
    # cross-reference case and the rule that a refusal still cites the clause
    # it relied on; v4 requires the contract to be named in the prose, not
    # only in the citations list, because "(Section 4.2)" does not say which
    # of five agreements it came from; v5 adds the injection-resistance
    # wording without losing v4's terminology answers. See docs/answer-eval.md.
    prompt_version: str
    # Presidio reports a confidence per detection. In this corpus a ZIP+4
    # mistaken for a social security number scored 0.05, while genuine
    # detections score 0.85 or above, so the threshold has room either side.
    masking_min_confidence: float
    masking_enabled: bool

    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"


class ConfigurationError(RuntimeError):
    """Raised when the environment does not describe a complete configuration."""


def _missing_variables(error: ValidationError) -> list[str]:
    """The environment variable names behind a pydantic 'field required' error."""
    return sorted(
        str(item["loc"][0]).upper()
        for item in error.errors()
        if item["type"] == "missing" and item.get("loc")
    )


@lru_cache
def get_settings() -> Settings:
    """Cached so the environment is read once per process."""
    try:
        return Settings()
    except ValidationError as error:
        missing = _missing_variables(error)
        if not missing:
            raise
        raise ConfigurationError(
            "Configuration is incomplete. These environment variables are not "
            "set:\n  " + "\n  ".join(missing) + "\n\n"
            "Every setting is read from the environment; the code holds no "
            "fallback values. Copy the template and fill it in:\n"
            "  copy .env.example .env        (Windows)\n"
            "  cp .env.example .env          (macOS, Linux)"
        ) from None


settings = get_settings()


def list_contracts() -> list[Path]:
    """Every contract PDF in the input folder.

    Matched case-insensitively on purpose: one of the five files ends in
    .PDF rather than .pdf, and a plain glob for '*.pdf' silently misses it.
    """
    if not settings.contracts_dir.is_dir():
        return []
    return sorted(
        p for p in settings.contracts_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )
