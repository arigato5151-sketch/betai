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
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade "pip==26.1.2" \
    && install_ok=0 \
    && for attempt in 1 2 3 4 5; do \
        python -m pip install --retries 10 --timeout 120 -r requirements.txt \
        && install_ok=1 \
        && break; \
    done \
    && [ "$install_ok" = "1" ] \
    && python -m pip cache purge
COPY backend/ ./
COPY --from=frontend-builder /frontend/dist ./frontend_dist

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
