"""
tmdb.py
-------
Optional enrichment: fetches TMDB metadata (poster URL, tagline, overview)
for each MovieLens movie and merges it into the movies DataFrame.

Requirements:
  - A free TMDB API key: https://www.themoviedb.org/settings/api
  - Set env var TMDB_API_KEY, or pass api_key= explicitly.

Usage:
    python -m data.tmdb                             # enriches data/raw/ml-100k/
    python -m data.tmdb --movies path/to/movies.csv
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMG_BASE   = "https://image.tmdb.org/t/p/w342"  # 342px-wide poster


def fetch_tmdb_metadata(
    title: str,
    year: int,
    api_key: str,
    session: requests.Session,
) -> dict:
    """Search TMDB for a single movie. Returns a dict of enriched fields."""
    params = {
        "api_key":       api_key,
        "query":         title,
        "year":          year if year > 0 else "",
        "language":      "en-US",
        "include_adult": False,
    }
    try:
        resp = session.get(TMDB_SEARCH_URL, params=params, timeout=8)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return {}
        hit = results[0]  # take top result
        poster = hit.get("poster_path")
        return {
            "tmdb_id":     hit.get("id"),
            "overview":    hit.get("overview", ""),
            "poster_url":  f"{TMDB_IMG_BASE}{poster}" if poster else "",
            "popularity":  hit.get("popularity", 0.0),
            "vote_avg":    hit.get("vote_average", 0.0),
        }
    except Exception as exc:
        return {"_error": str(exc)}


def enrich_movies(
    movies: pd.DataFrame,
    api_key: str | None = None,
    sleep_sec: float = 0.25,
    max_movies: int | None = None,
) -> pd.DataFrame:
    """
    Add TMDB columns to a movies DataFrame.

    Skips movies that already have a tmdb_id (so you can resume interrupted runs).
    Respects TMDB's free-tier rate limit (~40 req/s) via sleep_sec.

    Args:
        movies:     DataFrame with columns [item_id, title, year, ...].
        api_key:    TMDB v3 API key. Falls back to env var TMDB_API_KEY.
        sleep_sec:  Pause between requests (default 0.25s ≈ 4 req/s, well under limit).
        max_movies: Limit enrichment to the first N rows (useful for testing).

    Returns:
        Enriched DataFrame with extra columns: tmdb_id, overview, poster_url,
        popularity, vote_avg.
    """
    key = api_key or os.environ.get("TMDB_API_KEY")
    if not key:
        raise ValueError(
            "TMDB API key required. Set TMDB_API_KEY env var or pass api_key=."
        )

    df = movies.copy()
    for col in ["tmdb_id", "overview", "poster_url", "popularity", "vote_avg"]:
        if col not in df.columns:
            df[col] = None

    rows = df if max_movies is None else df.head(max_movies)
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    enriched = 0
    skipped  = 0
    for i, (idx, row) in enumerate(rows.iterrows()):
        if pd.notna(row.get("tmdb_id")):
            skipped += 1
            continue

        meta = fetch_tmdb_metadata(
            title=row["title"],
            year=int(row.get("year", 0)),
            api_key=key,
            session=session,
        )
        for col, val in meta.items():
            if col != "_error":
                df.at[idx, col] = val

        enriched += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}] enriched={enriched}, skipped={skipped}")

        time.sleep(sleep_sec)

    print(f"Done. {enriched} fetched, {skipped} already present.")
    return df


if __name__ == "__main__":
    import argparse
    from data.preprocess import load_movies, RAW_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--movies",    default=None, help="Path to movies CSV (optional)")
    parser.add_argument("--api-key",   default=None, help="TMDB API key (or set TMDB_API_KEY env var)")
    parser.add_argument("--max",       type=int, default=50, help="Max movies to enrich (default 50)")
    parser.add_argument("--out",       default=str(RAW_DIR / "movies_enriched.csv"))
    args = parser.parse_args()

    if args.movies:
        movies = pd.read_csv(args.movies)
    else:
        movies = load_movies()

    enriched = enrich_movies(movies, api_key=args.api_key, max_movies=args.max)
    enriched.to_csv(args.out, index=False)
    print(f"Saved to {args.out}")
    print(enriched[["title", "year", "poster_url", "overview"]].head())