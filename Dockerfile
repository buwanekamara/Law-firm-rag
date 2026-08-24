# syntax=docker/dockerfile:1

# builder  installs dependencies from the lock file
# models   downloads the embedding models into the image
# runtime  carries the venv, the models and the code - nothing else

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# /app, not /build: a venv hard-codes its own path into the scripts it
# generates, so it has to be built where it will run.
WORKDIR /app

# Dependencies first so editing code does not re-resolve the tree.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Masking needs a ~560MB spaCy model. Opt in with:
#   INSTALL_MASKING=true docker compose build
ARG INSTALL_MASKING=false
RUN if [ "$INSTALL_MASKING" = "true" ]; then \
        /app/.venv/bin/python -m spacy download en_core_web_lg; \
    fi

FROM builder AS models

# Baked into a layer, so these are build arguments, not runtime settings.
# They must match DENSE_MODEL and SPARSE_MODEL in .env.
ARG DENSE_MODEL=BAAI/bge-small-en-v1.5
ARG SPARSE_MODEL=Qdrant/bm25

ENV FASTEMBED_CACHE_PATH=/opt/models

# Instantiating each model downloads its weights, so startup does not.
RUN /app/.venv/bin/python -c "\
import sys; \
from fastembed import SparseTextEmbedding, TextEmbedding; \
TextEmbedding(model_name=sys.argv[1]); \
SparseTextEmbedding(model_name=sys.argv[2])" "${DENSE_MODEL}" "${SPARSE_MODEL}"

FROM python:3.12-slim-bookworm AS runtime

RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app

COPY --from=models --chown=app:app /app/.venv /app/.venv
COPY --from=models --chown=app:app /opt/models /opt/models
COPY --chown=app:app app ./app
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app contracts ./contracts

# Written by the ingest step.
RUN mkdir -p /app/data && chown -R app:app /app/data

# HF_HUB_OFFLINE makes a cache miss fail loudly instead of downloading 130MB.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/opt/models \
    HF_HUB_OFFLINE=1

USER app
EXPOSE 8000

# No curl or wget in a slim image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
