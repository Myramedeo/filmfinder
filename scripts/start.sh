#!/usr/bin/env bash
# scripts/start.sh
# ─────────────────────────────────────────────────────────────────────────────
# Entry-point used by the Dockerfile in place of a bare `uvicorn` call.
#
# Strategy
# ─────────
# We commit the trained checkpoint to the repo so that the Docker image
# is self-contained and cold-start is fast (~2 s on Render's free tier).
#
# This script is a safety net: if the checkpoint is missing for any reason
# (first deploy before training, or someone deleted it), it re-trains from
# scratch before starting the server.  Training takes ~2-3 min on CPU and
# only runs once per container lifecycle.
#
# Usage (Dockerfile CMD):
#   CMD ["bash", "scripts/start.sh"]

set -euo pipefail

CKPT="${MODEL_CHECKPOINT:-/app/models/checkpoints/best_model.pt}"

if [ ! -f "$CKPT" ]; then
  echo "════════════════════════════════════════════"
  echo " Checkpoint not found at $CKPT"
  echo " Running training pipeline (CPU, ~2-3 min)…"
  echo "════════════════════════════════════════════"

  # 1. Download data
  python -m data.download

  # 2. Train (fast settings for CI / first-deploy)
  python -m models.train \
    --epochs 10 \
    --batch-size 1024 \
    --output-dim 64 \
    --patience 3 \
    --device cpu

  echo "Training complete. Checkpoint saved to $CKPT"
else
  echo "Checkpoint found at $CKPT — skipping training."
fi

echo "Starting FastAPI server…"
exec uvicorn api.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers 1 \
  --timeout-keep-alive 30