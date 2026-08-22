"""Single source of truth for configuration.

Every tunable value in the project lives here: folder paths, model names,
retrieval knobs, feature switches. Values are read from environment variables
(or a local .env file); the defaults below apply when a variable is not set.

Nothing secret is hard-coded — the gateway key only ever arrives via the
environment.
"""

from functools import lru_cache
from pathlib import Path

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
    contracts_dir: Path = PROJECT_ROOT / "contracts"
    data_dir: Path = PROJECT_ROOT / "data"

    # --- LLM, via the Vercel AI Gateway ------------------------------------
    ai_gateway_api_key: str = ""
    ai_gateway_base_url: str = "https://ai-gateway.vercel.sh/v1"
    llm_model: str = "openai/gpt-4o-mini"
    # A different model judges the answers in the offline evaluation, so the
    # answering model is not grading its own homework.
    judge_model: str = "anthropic/claude-haiku-4.5"
    llm_temperature: float = 0.0

    # --- vector store ------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    # Set this instead of running a server and Qdrant runs embedded, inside
    # this process, storing to the given folder. See app.indexing.get_client.
    qdrant_path: str = ""
    qdrant_collection: str = "contract_chunks"
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"

    # --- retrieval ---------------------------------------------------------
    # "dense", "sparse" or "hybrid". Hybrid is the measured default: dense
    # alone misses the redacted-price and placeholder-date clauses entirely.
    # See the docstring at the top of app/retrieval.py and docs/retrieval-eval.md.
    search_mode: str = "hybrid"
    # 8, not 5: measured. At 5 the trademark licence's Section 4.4 (an
    # automatic termination trigger) fell one slot outside the window and the
    # answer silently omitted it. See docs/retrieval-eval.md.
    top_k: int = 8
    # Relevance gate. If the closest chunk's DENSE cosine similarity is below
    # this, the question is refused before the model is called at all. Not the
    # fused hybrid score: RRF ranks by agreement between two result lists, so
    # the top hit of a nonsense query scores about as well as the top hit of a
    # good one. 0 disables the gate - run scripts/calibrate_gate.py to pick a
    # value from data rather than guessing one.
    min_score: float = 0.0
    # Give the model one chance to correct a citation that names a section it
    # was never shown, before the citation is dropped.
    citation_retry: bool = True

    # --- prompts and guards ------------------------------------------------
    # v4 is the measured default. v1 mishandled every "the value is not here"
    # case; v2 added the redaction and placeholder rules; v3 added the
    # cross-reference case and the rule that a refusal still cites the clause
    # it relied on; v4 requires the contract to be named in the prose, not
    # only in the citations list, because "(Section 4.2)" does not say which
    # of five agreements it came from. v3 and v4 both score 21/22 but fail
    # different questions; v4 was chosen for stability and readability.
    # See docs/answer-eval.md.
    prompt_version: str = "v5"
    masking_enabled: bool = False

    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"


@lru_cache
def get_settings() -> Settings:
    """Cached so the .env file is parsed once per process."""
    return Settings()


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
