# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --upgrade pip --no-cache-dir

COPY requirements.txt .
# CPU-only torch keeps the image small (~750 MB vs ~3 GB for CUDA)
RUN pip install --no-cache-dir torch==2.2.0+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install --no-cache-dir -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser /app
USER appuser

EXPOSE $PORT

CMD uvicorn api.main:app \
        --host 0.0.0.0 \
        --port $PORT \
        --workers 1 \
        --timeout-keep-alive 30