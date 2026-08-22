# syntax=docker/dockerfile:1

# Three stages, for three different reasons.
#
#   builder  installs dependencies with uv, from the lock file, so a build is
#            reproducible rather than "whatever PyPI served that day"
#   models   downloads the two embedding models into the image, so the first
#            question after startup is fast and needs no network
#   runtime  carries the virtual environment, the models and the code - and
#            neither uv, nor a compiler, nor a package cache
#
# The result runs as a non-root user and, once built, never reaches the
# network except to call the model gateway.

# --- builder ---------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# /app, not /build. A virtual environment records its own absolute path in the
# scripts it generates, so one assembled at /build and copied to /app has a
# uvicorn that points at a directory the final image does not have. Building it
# where it will live avoids that entirely.
WORKDIR /app

# Dependencies first, project second. Editing application code then rebuilds
# only the last layer instead of resolving the dependency tree again.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Personal-data masking needs a spaCy language model that the lock file cannot
# carry, and it is ~560MB. Masking ships switched off, so the model is opt-in:
#
#   INSTALL_MASKING=true docker compose build
#
# Left out, the image is smaller and MASKING_ENABLED=true reports plainly that
# the model is absent rather than failing halfway through a question.
ARG INSTALL_MASKING=false
RUN if [ "$INSTALL_MASKING" = "true" ]; then \
        /app/.venv/bin/python -m spacy download en_core_web_lg; \
    fi

# --- models ----------------------------------------------------------------
FROM builder AS models

# The weights are baked into a layer, so these are build arguments rather than
# runtime settings: changing either one means rebuilding the image, and the
# stored index has to be rebuilt with it. They must match DENSE_MODEL and
# SPARSE_MODEL in .env.
ARG DENSE_MODEL=BAAI/bge-small-en-v1.5
ARG SPARSE_MODEL=Qdrant/bm25

ENV FASTEMBED_CACHE_PATH=/opt/models

# Instantiating each model downloads its weights. Doing it at build time means
# startup does not wait on a 130MB download, and a deployment behind a firewall
# still works.
RUN /app/.venv/bin/python -c "\
import sys; \
from fastembed import SparseTextEmbedding, TextEmbedding; \
TextEmbedding(model_name=sys.argv[1]); \
SparseTextEmbedding(model_name=sys.argv[2])" "${DENSE_MODEL}" "${SPARSE_MODEL}"

# --- runtime ---------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# Not root. A question-answering service has no business owning its own files.
RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app

COPY --from=models --chown=app:app /app/.venv /app/.venv
COPY --from=models --chown=app:app /opt/models /opt/models
COPY --chown=app:app app ./app
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app contracts ./contracts

# data/ holds extracted text and chunk files, written by the ingest step.
RUN mkdir -p /app/data && chown -R app:app /app/data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/opt/models \
    HF_HUB_OFFLINE=1

USER app
EXPOSE 8000

# The image is slim: no curl, no wget. Python is already here.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
