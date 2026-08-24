"""Every setting, in one place.

Values come from the environment or .env. Nothing has a default in code, so a
missing variable fails at startup instead of running on a value nobody chose.
The two folder paths are the exception - they follow this file, so the project
works the same in a checkout and in a container.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Derived, not configured - see the module docstring.
    contracts_dir: Path = PROJECT_ROOT / "contracts"
    data_dir: Path = PROJECT_ROOT / "data"

    ai_gateway_api_key: str
    ai_gateway_base_url: str
    # Any id the gateway accepts - uv run scripts/ask.py --models lists them.
    llm_model: str
    # Grades answers in eval/. Keep it a different vendor from llm_model, so
    # the answering model is not marking its own work.
    judge_model: str
    # 0.0-2.0. Leave at 0 for evaluation; higher makes the same question score
    # differently run to run.
    llm_temperature: float
    llm_timeout_seconds: float

    # A server URL, or ":memory:" for a throwaway index.
    qdrant_url: str
    # Set instead of qdrant_url to run Qdrant embedded in this process - one
    # process at a time. See app/search/indexing.get_client.
    qdrant_path: str
    # Change to keep several indexes side by side and switch between them.
    qdrant_collection: str
    # Any fastembed model id. Changing either invalidates the stored index:
    # re-run scripts/index.py.
    dense_model: str
    sparse_model: str

    # "dense" (meaning), "sparse" (exact words) or "hybrid" (both, fused by
    # rank). eval/retrieval_eval.py compares all three - docs/retrieval-eval.md.
    search_mode: str
    # Excerpts sent to the model, 1-20. Lower drops sections that answer part
    # of a question; higher dilutes the prompt and costs more.
    top_k: int
    # Refuse below this DENSE cosine similarity, before the model is called.
    # Not the fused score - RRF ranks by agreement, so a nonsense query's top
    # hit scores about as well as a good one's. 0 = off; pick a value with
    # scripts/calibrate_gate.py rather than by guessing.
    min_score: float
    # Let the model correct an invented citation once before it is dropped.
    # False is one call cheaper and shows the raw failure rate.
    citation_retry: bool
    # 1-10. Each half of a hybrid search fetches this multiple of top_k before
    # fusion. Below 2 there is little for the fusion to work with.
    prefetch_multiplier: int

    # 100-400 words. Above ~380 the embedding model truncates the tail of a
    # clause; below ~150 sections split that should stay whole. Overlap is how
    # much of the previous piece a split repeats.
    # Changing either: re-run extract.py, chunk.py, index.py.
    chunk_max_words: int
    chunk_overlap_words: int
    # Chunks embedded at once. Affects memory and speed, nothing else.
    embed_batch_size: int

    # 0-20 previous exchanges carried into a follow-up. 0 disables follow-ups.
    history_turns: int

    # Which app/generate/prompts/answer_*.md file to use. A new file appears
    # here with no code change - eval/answer_eval.py compares versions.
    prompt_version: str
    # 0.0-1.0. Presidio's confidence floor. Below 0.6 a ZIP+4 is masked as a
    # social security number; above ~0.9 real names slip through.
    masking_min_confidence: float
    masking_enabled: bool

    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"


class ConfigurationError(RuntimeError):
    """The environment does not describe a complete configuration."""


def _missing_variables(error: ValidationError) -> list[str]:
    return sorted(
        str(item["loc"][0]).upper()
        for item in error.errors()
        if item["type"] == "missing" and item.get("loc")
    )


@lru_cache
def get_settings() -> Settings:
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
            "fallback values. Add the missing lines to the .env file in the "
            "project root, or set them in the environment."
        ) from None


settings = get_settings()


def list_contracts() -> list[Path]:
    """Every contract PDF in the input folder.

    Case-insensitive: one of the five files ends .PDF, and a plain '*.pdf'
    glob silently misses it.
    """
    if not settings.contracts_dir.is_dir():
        return []
    return sorted(
        p for p in settings.contracts_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )
