"""
preprocess.py
-------------
Loads raw MovieLens 100K files and produces clean, model-ready artefacts:

  - ratings DataFrame        (user_idx, item_idx, rating, timestamp)
  - movies DataFrame         (item_idx, title, year, genre_vec)
  - user2idx / item2idx      contiguous integer ID maps
  - genre vocabulary         list of genre names (20 genres in ML-100K)
  - train / val / test splits (time-based)

Design notes
------------
Two-tower models need:
  * User tower:  user_idx  →  learnable embedding
  * Item tower:  item_idx + genre_vec  →  embedding fused with content
  * Label:       binary (rating ≥ 4 = positive) for BPR/BCE loss,
                 or raw rating for MSE loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent / "raw" / "ml-100k"

# MovieLens 100K genre columns (order matches the u.item bitmask)
DEFAULT_GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]
GENRES = DEFAULT_GENRES.copy()


def _genre_vocab_from_movies(movies_df: pd.DataFrame) -> list[str]:
    """Infer the genre vocabulary from the raw movie metadata."""
    raw_genres = movies_df["genres"].astype(str).fillna("").str.split("|")
    unique = sorted({genre for row in raw_genres for genre in row if genre})
    return unique or DEFAULT_GENRES.copy()


def _build_genre_vector(genres: str, vocab: list[str]) -> np.ndarray:
    """Create a multi-hot genre vector for either 100K or 25M-style metadata."""
    present = set(genres.split("|")) if isinstance(genres, str) else set()
    return np.asarray([1.0 if genre in present else 0.0 for genre in vocab], dtype=np.float32)


# ---------------------------------------------------------------------------
# Raw loaders
# ---------------------------------------------------------------------------

def load_ratings(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load ratings from either 100K or 25M-style raw data."""
    ratings_csv = raw_dir / "ratings.csv"
    if ratings_csv.exists():
        df = pd.read_csv(
            ratings_csv,
            usecols=["userId", "movieId", "rating", "timestamp"],
            dtype={"userId": int, "movieId": int, "rating": float, "timestamp": int},
        )
        df = df.rename(columns={"userId": "user_id", "movieId": "item_id"})
        return df.sort_values("timestamp").reset_index(drop=True)

    df = pd.read_csv(
        raw_dir / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": int, "item_id": int, "rating": float, "timestamp": int},
    )
    return df.sort_values("timestamp").reset_index(drop=True)


def load_movies(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load movies from either 100K or 25M-style raw data."""
    global GENRES

    movies_csv = raw_dir / "movies.csv"
    if movies_csv.exists():
        df = pd.read_csv(
            movies_csv,
            usecols=["movieId", "title", "genres"],
            dtype={"movieId": int},
        )
        df = df.rename(columns={"movieId": "item_id"})
        genre_vocab = _genre_vocab_from_movies(df)
        GENRES = genre_vocab.copy()

        df["year"] = (
            df["title"]
            .str.extract(r"\((\d{4})\)")
            .astype(float)
            .fillna(0)
            .astype(int)
        )
        y_min, y_max = df["year"].replace(0, np.nan).min(), df["year"].max()
        df["year_norm"] = ((df["year"] - y_min) / (y_max - y_min + 1e-8)).fillna(0.0)
        df["genre_vec"] = [
            _build_genre_vector(row["genres"], GENRES).tolist() for _, row in df.iterrows()
        ]
        return df[["item_id", "title", "year", "year_norm", "genre_vec"]]

    cols = ["item_id", "title", "release_date", "video_date", "imdb_url"] + DEFAULT_GENRES
    df = pd.read_csv(
        raw_dir / "u.item",
        sep="|",
        names=cols,
        encoding="latin-1",
        usecols=["item_id", "title", "release_date"] + DEFAULT_GENRES,
    )
    GENRES = DEFAULT_GENRES.copy()

    # Extract 4-digit year from title string, e.g. "Toy Story (1995)" → 1995
    df["year"] = (
        df["title"]
        .str.extract(r"\((\d{4})\)")
        .astype(float)
        .fillna(0)
        .astype(int)
    )

    # Normalise year to [0, 1] relative to dataset range
    y_min, y_max = df["year"].replace(0, np.nan).min(), df["year"].max()
    df["year_norm"] = ((df["year"] - y_min) / (y_max - y_min + 1e-8)).fillna(0.0)

    # genre_vec: (n_movies, 19) float32 matrix stored per-row as list
    genre_matrix = df[GENRES].values.astype(np.float32)
    df["genre_vec"] = list(genre_matrix) # type: ignore

    return df[["item_id", "title", "year", "year_norm", "genre_vec"]]


def load_users(raw_dir: Path = RAW_DIR, ratings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load user metadata or synthesize defaults for 25M-style datasets."""
    users_csv = raw_dir / "users.csv"
    if users_csv.exists():
        df = pd.read_csv(users_csv, usecols=["userId", "age", "gender", "occupation"])
        df = df.rename(columns={"userId": "user_id"})
        df["gender_m"] = (df["gender"] == "M").astype(np.float32)
        occ_map = {o: i for i, o in enumerate([
            "administrator", "artist", "doctor", "educator", "engineer",
            "entertainment", "executive", "healthcare", "homemaker", "lawyer",
            "librarian", "marketing", "none", "other", "programmer",
            "retired", "salesman", "scientist", "student", "technician", "writer",
        ])}
        df["occ_idx"] = df["occupation"].map(occ_map).fillna(12).astype(int)
        age_min, age_max = df["age"].min(), df["age"].max()
        df["age_norm"] = ((df["age"] - age_min) / (age_max - age_min + 1e-8)).astype(np.float32)
        return df[["user_id", "age_norm", "gender_m", "occ_idx"]]

    if ratings is None:
        ratings = load_ratings(raw_dir)

    users = pd.DataFrame({"user_id": sorted(ratings["user_id"].unique())})
    users["age_norm"] = 0.5
    users["gender_m"] = 0.5
    users["occ_idx"] = 0
    return users


# ---------------------------------------------------------------------------
# ID mapping helpers
# ---------------------------------------------------------------------------

def build_id_maps(ratings: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    """
    Return (user2idx, idx2user, item2idx, idx2item) for all IDs seen in ratings.
    Indices are contiguous integers starting at 0.
    """
    users = sorted(ratings["user_id"].unique())
    items = sorted(ratings["item_id"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {m: i for i, m in enumerate(items)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: m for m, i in item2idx.items()}
    return user2idx, idx2user, item2idx, idx2item


# ---------------------------------------------------------------------------
# Train / val / test split  (time-based — no leakage)
# ---------------------------------------------------------------------------

def time_split(
    ratings: pd.DataFrame,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split ratings chronologically.
    The last (val_frac + test_frac) of interactions form val + test.
    This mirrors real-world evaluation: train on the past, predict the future.
    """
    n = len(ratings)
    val_start  = int(n * (1 - val_frac - test_frac))
    test_start = int(n * (1 - test_frac))

    train = ratings.iloc[:val_start].copy()
    val   = ratings.iloc[val_start:test_start].copy()
    test  = ratings.iloc[test_start:].copy()
    return train, val, test


# ---------------------------------------------------------------------------
# Label binarisation
# ---------------------------------------------------------------------------

def binarise(
    ratings: pd.DataFrame,
    threshold: float = 4.0,
    task: Literal["binary", "regression"] = "binary",
) -> pd.DataFrame:
    """
    Add a `label` column.
      - binary     → 1 if rating ≥ threshold else 0
      - regression → label = rating / 5.0  (normalised)
    """
    df = ratings.copy()
    if task == "binary":
        df["label"] = (df["rating"] >= threshold).astype(np.float32)
    else:
        df["label"] = (df["rating"] / 5.0).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_dataset(
    raw_dir: Path = RAW_DIR,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    task: Literal["binary", "regression"] = "binary",
    label_threshold: float = 4.0,
) -> dict:
    """
    Full ingestion pipeline. Returns a dict with all artefacts needed for training:

    {
      "train":      pd.DataFrame,   # (user_idx, item_idx, label, ...)
      "val":        pd.DataFrame,
      "test":       pd.DataFrame,
      "movies":     pd.DataFrame,   # (item_idx, title, year_norm, genre_vec)
      "users":      pd.DataFrame,   # (user_idx, age_norm, gender_m, occ_idx)
      "user2idx":   dict,
      "item2idx":   dict,
      "idx2user":   dict,
      "idx2item":   dict,
      "n_users":    int,
      "n_items":    int,
      "n_genres":   int,
    }
    """
    print("Loading raw data …")
    ratings = load_ratings(raw_dir)
    movies  = load_movies(raw_dir)
    users   = load_users(raw_dir, ratings=ratings)

    print("Building ID maps …")
    user2idx, idx2user, item2idx, idx2item = build_id_maps(ratings)

    # Map original IDs → contiguous indices in the ratings frame
    ratings["user_idx"] = ratings["user_id"].map(user2idx)
    ratings["item_idx"] = ratings["item_id"].map(item2idx)
    ratings = binarise(ratings, threshold=label_threshold, task=task)

    print("Splitting train / val / test …")
    train, val, test = time_split(ratings, val_frac=val_frac, test_frac=test_frac)

    # Attach contiguous indices to look-up tables too
    movies["item_idx"] = movies["item_id"].map(item2idx)
    movies = movies.dropna(subset=["item_idx"])
    movies["item_idx"] = movies["item_idx"].astype(int)

    users["user_idx"] = users["user_id"].map(user2idx)
    users = users.dropna(subset=["user_idx"])
    users["user_idx"] = users["user_idx"].astype(int)

    stats = {
        "n_users":  len(user2idx),
        "n_items":  len(item2idx),
        "n_genres": len(GENRES),
    }

    print(
        f"\nDataset summary\n"
        f"  users  : {stats['n_users']:,}\n"
        f"  items  : {stats['n_items']:,}\n"
        f"  train  : {len(train):,} interactions\n"
        f"  val    : {len(val):,} interactions\n"
        f"  test   : {len(test):,} interactions\n"
        f"  task   : {task}  (threshold={label_threshold})\n"
        f"  pos %  : {train['label'].mean():.1%} of train are positive\n"
    )

    return {
        "train": train,
        "val":   val,
        "test":  test,
        "movies":   movies,
        "users":    users,
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2user": idx2user,
        "idx2item": idx2item,
        **stats,
    }


if __name__ == "__main__":
    from download import download_movielens
    download_movielens()
    ds = build_dataset()
    print("Sample training rows:")
    print(ds["train"][["user_idx", "item_idx", "rating", "label"]].head())