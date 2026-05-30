"""
schemas.py
----------
Pydantic v2 models for all request / response payloads.

"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class MovieOut(BaseModel):
    """A single recommended movie returned to the client."""
    movie_id:   int           = Field(..., description="Internal MovieLens item ID")
    title:      str           = Field(..., description="Movie title including release year")
    year:       int           = Field(..., description="Release year (0 = unknown)")
    genres:     list[str]     = Field(..., description="Genre tags for this movie")
    score:      float         = Field(..., description="Cosine similarity score in [0, 1]")
    poster_url: str | None    = Field(None, description="TMDB poster URL (if enriched)")
    overview:   str | None    = Field(None, description="TMDB plot overview (if enriched)")


# ---------------------------------------------------------------------------
# POST /recommend
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    """
    Request body for /recommend.

    Two usage modes:
      1. Known user  → supply user_id; the model looks up the trained embedding.
      2. Cold start  → supply liked_movie_ids; the model averages their item
                       vectors to approximate a user preference vector.

    At least one of user_id or liked_movie_ids must be provided.
    """
    user_id:        int | None  = Field(None,  description="Existing user ID (1–943 for ML-100K)")
    liked_movie_ids: list[int]  = Field(default_factory=list,
                                        description="Cold-start: list of movie IDs the user likes")
    top_k:          int         = Field(10,    ge=1, le=100, description="Number of results to return")
    exclude_seen:   bool        = Field(True,  description="Filter out movies the user has already rated")

    @field_validator("liked_movie_ids")
    @classmethod
    def max_liked(cls, v: list[int]) -> list[int]:
        if len(v) > 50:
            raise ValueError("liked_movie_ids may contain at most 50 movie IDs")
        return v

    def model_post_init(self, __context) -> None:  # noqa: ANN001
        if self.user_id is None and not self.liked_movie_ids:
            raise ValueError("Provide at least one of: user_id, liked_movie_ids")


class RecommendResponse(BaseModel):
    """Response envelope for /recommend."""
    user_id:      int | None    = Field(None, description="Echo of requested user_id")
    mode:         Literal["known_user", "cold_start"]
    results:      list[MovieOut]
    model_version: str          = Field(..., description="Checkpoint identifier")


# ---------------------------------------------------------------------------
# GET /movies/{movie_id}
# ---------------------------------------------------------------------------

class MovieDetailResponse(BaseModel):
    """Full movie record."""
    movie_id:   int
    title:      str
    year:       int
    genres:     list[str]
    poster_url: str | None = None
    overview:   str | None = None


# ---------------------------------------------------------------------------
# GET /similar/{movie_id}
# ---------------------------------------------------------------------------

class SimilarRequest(BaseModel):
    top_k: int = Field(10, ge=1, le=50)


class SimilarResponse(BaseModel):
    """Item-to-item similarity results."""
    seed_movie:   MovieDetailResponse
    similar:      list[MovieOut]
    model_version: str


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status:        Literal["ok", "degraded"]
    model_loaded:  bool
    n_users:       int
    n_items:       int
    model_version: str