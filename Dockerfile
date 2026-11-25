# Image de l'API de prédiction. Le modèle (models/cnn.pt + models/meta.json) est embarqué : le conteneur
# n'a besoin ni de MLflow, ni du S3, ni d'aucun secret.
#
#   docker build --build-arg GIT_SHA=$(git rev-parse HEAD) -t imu-api:dev .
#   docker run --rm -p 8000:8000 imu-api:dev        # http://localhost:8000/app/

FROM python:3.12-slim

# Version d'uv fixée : celle qui a produit uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 1. Dépendances seules : cette couche (la plus lourde, torch) reste en cache tant que uv.lock ne change pas.
#    --locked : les versions exactes de uv.lock ; --no-default-groups : ni MLflow, ni DVC, ni outils de dev.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-default-groups --no-install-project

# 2. Le code, qui change à chaque commit.
COPY src ./src
RUN uv sync --locked --no-default-groups --no-editable && rm -rf build

# 3. Le modèle, et la version qu'il annoncera dans chaque réponse (le commit qui a produit l'image).
COPY models/cnn.pt models/meta.json ./models/
ARG GIT_SHA=dev
ENV MODEL_URI=/app/models \
    MODEL_VERSION=${GIT_SHA} \
    PATH="/app/.venv/bin:$PATH"

# Utilisateur sans privilèges.
RUN useradd --create-home --uid 10001 appuser && mkdir /app/logs && chown appuser /app/logs
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "imu.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
