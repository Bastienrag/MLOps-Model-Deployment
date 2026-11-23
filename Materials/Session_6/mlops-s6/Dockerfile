#   uv run python -m fraud.export_model   # Registry @staging -> ./model
#   docker build -t fraud-api:dev .

FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-default-groups --no-install-project

COPY README.md params.yaml ./
COPY src ./src
RUN uv sync --locked --no-default-groups --no-editable

COPY model ./model

RUN useradd --create-home appuser && mkdir -p /app/logs && chown appuser /app/logs
USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_URI=/app/model
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "fraud.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
