FROM node:20-alpine AS frontend-builder

WORKDIR /frontend
ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FRONTEND_DIST_DIR=/app/frontend_dist

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && groupadd --gid 10001 betai \
    && useradd --uid 10001 --gid betai --no-create-home --shell /usr/sbin/nologin betai \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./
COPY scripts/download_binary_wheel.py /tmp/download_binary_wheel.py
RUN python -m pip install --no-cache-dir --upgrade "pip==26.1.2" \
    && python /tmp/download_binary_wheel.py catboost 1.2.10 /tmp/wheels \
    && python -m pip install --no-deps /tmp/wheels/catboost-*.whl \
    && install_ok=0 \
    && for attempt in 1 2 3 4 5; do \
        if python -m pip install --retries 10 --resume-retries 20 --timeout 120 -r requirements.txt; then \
            install_ok=1; \
            break; \
        fi; \
    done \
    && [ "$install_ok" = "1" ] \
    && python -m pip cache purge
ENV HOME=/tmp
ENV MPLCONFIGDIR=/tmp/matplotlib
COPY --chown=betai:betai backend/ ./
COPY --chown=betai:betai --from=frontend-builder /frontend/dist ./frontend_dist
RUN mkdir -p /app/artifacts/models /var/run/celery /tmp/matplotlib \
    && chown -R betai:betai /app/artifacts /var/run/celery /tmp/matplotlib

USER betai

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
