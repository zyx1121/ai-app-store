# The portability contract for a Linux host. Version 1 of the platform runs the
# app as a native process, this image is what CI builds on publish.
# No model and no Postgres inside: both are injected as URLs.

FROM ghcr.io/astral-sh/uv:python3.13-alpine
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
COPY . .
ENV PORT=8000
EXPOSE $PORT
CMD ["uv", "run", "python", "main.py"]
