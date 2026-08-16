# Multi-stage production build using official Python 3.12 slim
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project specification
COPY pyproject.toml ./

# Install dependencies into virtual environment
RUN uv venv /app/.venv && \
    UV_HTTP_TIMEOUT=120 /bin/uv pip install --no-cache --python /app/.venv/bin/python -r pyproject.toml

# Final runtime image
FROM python:3.12-slim AS runtime

WORKDIR /app

# Create non-root system user
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

# Copy virtualenv and code
COPY --from=builder /app/.venv /app/.venv
COPY src/ /app/src/
COPY docs/ /app/docs/
COPY pyproject.toml README.md LICENSE LEGAL.md /app/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HOST=0.0.0.0

USER appuser

EXPOSE 8080

CMD ["python", "-m", "src.server"]
