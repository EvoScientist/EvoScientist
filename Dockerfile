# syntax=docker/dockerfile:1.7

# ---------- Builder ----------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /src

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev \
        --extra all-channels

COPY EvoScientist ./EvoScientist
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable \
        --extra all-channels

# ---------- Runtime ----------
FROM python:3.11-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        tini \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node:24-bookworm-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:24-bookworm-slim /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

COPY --from=ghcr.io/astral-sh/uv:python3.11-bookworm-slim /usr/local/bin/uv /usr/local/bin/uv

ARG UID=1000
ARG GID=1000
RUN groupadd --gid ${GID} evosci \
    && useradd  --uid ${UID} --gid ${GID} --create-home --shell /bin/bash evosci

COPY --from=builder --chown=${UID}:${GID} /opt/venv /opt/venv

ENV PATH="/home/evosci/.evoscientist/.local/bin:/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EVOSCIENTIST_WORKSPACE_DIR=/workspace \
    EVOSCIENTIST_DATA_DIR=/home/evosci/.evoscientist \
    XDG_CONFIG_HOME=/home/evosci/.evoscientist/.config \
    UV_TOOL_DIR=/home/evosci/.evoscientist/.local/share/uv/tools \
    UV_TOOL_BIN_DIR=/home/evosci/.evoscientist/.local/bin

RUN mkdir -p /workspace \
        /home/evosci/.evoscientist/.config/evoscientist \
        /home/evosci/.evoscientist/.local/bin \
        /home/evosci/.evoscientist/.local/share/uv/tools \
    && chown -R ${UID}:${GID} /workspace /home/evosci

USER evosci
WORKDIR /workspace

LABEL org.opencontainers.image.title="EvoScientist" \
      org.opencontainers.image.description="EvoScientist agent with core + all-channels dependencies pre-installed." \
      org.opencontainers.image.source="https://github.com/EvoScientist/EvoScientist" \
      org.opencontainers.image.documentation="https://github.com/EvoScientist/EvoScientist#-docker" \
      org.opencontainers.image.licenses="Apache-2.0"

ENTRYPOINT ["tini", "--", "evosci"]
