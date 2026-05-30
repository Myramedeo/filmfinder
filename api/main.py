"""
main.py
-------
FastAPI application for the movie recommender.

Endpoints
---------
  GET  /health                 — liveness / readiness probe
  POST /recommend              — top-K recommendations for a user
  GET  /movies/{movie_id}      — single movie detail lookup
  GET  /similar/{movie_id}     — item-to-item similarity
  GET  /users                  — list of known user IDs (for the demo UI)

Run locally
-----------
    uvicorn api.main:app --reload --port 8000

Visit http://localhost:8000/docs for interactive API docs.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import (
    HealthResponse,
    MovieDetailResponse,
    MovieOut,
    RecommendRequest,
    RecommendResponse,
    SimilarResponse,
)
from model_store import store, ScoredItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scored_to_movie_out(item: ScoredItem) -> MovieOut:
    return MovieOut(
        movie_id   = item.movie_id,
        title      = item.title,
        year       = item.year,
        genres     = item.genres,
        score      = round(item.score, 4),
        poster_url = item.poster_url,
        overview   = item.overview,
    )


def _scored_to_detail(item: ScoredItem) -> MovieDetailResponse:
    return MovieDetailResponse(
        movie_id   = item.movie_id,
        title      = item.title,
        year       = item.year,
        genres     = item.genres,
        poster_url = item.poster_url,
        overview   = item.overview,
    )


# ---------------------------------------------------------------------------
# Lifespan: load model once at startup, clean up on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = os.environ.get("MODEL_CHECKPOINT")
    store.load(ckpt_path=Path(ckpt) if ckpt else None)
    yield
    # nothing to clean up for a CPU model


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Movie Recommender API",
    description=(
        "Two-tower neural recommender trained on MovieLens 100K.\n\n"
        "**Modes**\n"
        "- *Known user*: pass `user_id` to use the trained user embedding.\n"
        "- *Cold start*: pass `liked_movie_ids` to get recommendations based on movie taste.\n\n"
        "Trained with PyTorch; embeddings retrieved via dot-product similarity."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# CORS — allow the Streamlit / React frontend to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Simple in-process rate limiter: max 60 requests / 60s per IP
_rate_store: dict[str, list[float]] = {}
RATE_LIMIT   = int(os.environ.get("RATE_LIMIT",    "60"))
RATE_WINDOW  = int(os.environ.get("RATE_WINDOW_S", "60"))

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    ip  = request.client.host if request.client else "unknown"
    now = time.time()
    hits = [t for t in _rate_store.get(ip, []) if now - t < RATE_WINDOW]
    if len(hits) >= RATE_LIMIT:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": f"Rate limit exceeded: {RATE_LIMIT} requests / {RATE_WINDOW}s"},
        )
    hits.append(now)
    _rate_store[ip] = hits
    return await call_next(request)


# Request timing header (useful for debugging Render cold starts)
@app.middleware("http")
async def add_timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness + readiness probe. Returns 503 if the model has not loaded."""
    if not store.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not yet loaded",
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        n_users=store.n_users,
        n_items=store.n_items,
        model_version=store.model_version,
    )


@app.post("/recommend", response_model=RecommendResponse, tags=["recommend"])
def recommend(body: RecommendRequest):
    """
    Return top-K movie recommendations.

    **Known user** (fastest): `{"user_id": 42, "top_k": 10}`

    **Cold start**: `{"liked_movie_ids": [50, 172, 181], "top_k": 10}`
    """
    if not store.is_ready:
        raise HTTPException(503, "Model not loaded")

    # Validate user_id exists in training set
    if body.user_id is not None and body.user_id not in store._user2idx:
        raise HTTPException(
            status_code=404,
            detail=f"user_id {body.user_id} not found. "
                   f"Valid range: {min(store._user2idx)} – {max(store._user2idx)}",
        )

    try:
        results, mode = store.recommend(
            user_id=body.user_id,
            liked_movie_ids=body.liked_movie_ids,
            top_k=body.top_k,
            exclude_seen=body.exclude_seen,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RecommendResponse(
        user_id=body.user_id,
        mode=mode,
        results=[_scored_to_movie_out(r) for r in results],
        model_version=store.model_version,
    )


@app.get("/movies/{movie_id}", response_model=MovieDetailResponse, tags=["movies"])
def get_movie(movie_id: int):
    """Fetch metadata for a single movie by its MovieLens item ID."""
    if not store.is_ready:
        raise HTTPException(503, "Model not loaded")

    item = store.get_movie(movie_id)
    if item is None:
        raise HTTPException(404, f"movie_id {movie_id} not found")

    return _scored_to_detail(item)


@app.get("/similar/{movie_id}", response_model=SimilarResponse, tags=["movies"])
def similar_movies(movie_id: int, top_k: int = 10):
    """
    Item-to-item similarity: find movies whose learned embedding is
    closest to the seed movie in the shared vector space.
    """
    if not store.is_ready:
        raise HTTPException(503, "Model not loaded")

    seed = store.get_movie(movie_id)
    if seed is None:
        raise HTTPException(404, f"movie_id {movie_id} not found")

    try:
        similar = store.similar(movie_id, top_k=top_k)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    return SimilarResponse(
        seed_movie=_scored_to_detail(seed),
        similar=[_scored_to_movie_out(r) for r in similar],
        model_version=store.model_version,
    )


@app.get("/users", tags=["ops"])
def list_users(limit: int = 50):
    """
    Return a sample of known user IDs.
    Used by the demo frontend to populate the user picker.
    """
    if not store.is_ready:
        raise HTTPException(503, "Model not loaded")
    ids = [int(uid) for uid in store.known_user_ids()[:limit]]
    return {"user_ids": ids, "total": store.n_users}