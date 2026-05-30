"""
model_store.py
--------------
Singleton that owns the trained model and all inference state.

Responsibilities
----------------
Load the checkpoint produced by models/train.py on startup.
Pre-compute the full item embedding matrix once (all 1 682 movies).
Expose `recommend()` and `similar()` methods used by the API routes.
Handle both KNOWN-USER and COLD-START inference paths.

Keeping inference logic here (not in main.py) means the API routes
stay thin and the model can be swapped / mocked in tests without touching
the routing layer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import torch
import pandas as pd

# Make project root importable when this module is imported from anywhere
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.two_tower import TwoTowerModel, TwoTowerConfig
from data.preprocess import build_dataset, GENRES

# Default checkpoint path; override with env var MODEL_CHECKPOINT
DEFAULT_CKPT = ROOT / "models" / "checkpoints" / "best_model.pt"


# ---------------------------------------------------------------------------
# Typed result
# ---------------------------------------------------------------------------

class ScoredItem(NamedTuple):
    movie_id:   int
    title:      str
    year:       int
    genres:     list[str]
    score:      float
    poster_url: str | None
    overview:   str | None


# ---------------------------------------------------------------------------
# ModelStore
# ---------------------------------------------------------------------------

class ModelStore:
    """
    Loaded once at API startup via the FastAPI lifespan hook.

    Attributes exposed to routes
    ----------------------------
    model_version : str   — filename of the loaded checkpoint
    n_users       : int
    n_items       : int
    """

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def load(self, ckpt_path: Path | None = None) -> None:
        """Load checkpoint and pre-compute item index. Call once at startup."""
        ckpt_path = Path(ckpt_path or os.environ.get("MODEL_CHECKPOINT", DEFAULT_CKPT))

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt_path}. "
                "Run `python -m models.train` first, or set MODEL_CHECKPOINT env var."
            )

        self._device = torch.device("cpu")  # CPU is fine for inference at this scale

        print(f"[ModelStore] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)

        # Rebuild model from saved config
        cfg: TwoTowerConfig = ckpt["cfg"]
        self._model = TwoTowerModel(cfg).to(self._device)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()

        # ID maps
        self._user2idx: dict[int, int] = ckpt["user2idx"]
        self._item2idx: dict[int, int] = ckpt["item2idx"]
        self._idx2item: dict[int, int] = ckpt["idx2item"]

        # Rebuild movie metadata (needed for genre/title/year lookups)
        print("[ModelStore] Rebuilding movie metadata …")
        self._ds = build_dataset()
        movies_df = self._ds["movies"].set_index("item_idx")
        self._movies_df = movies_df

        # Pre-compute item embedding matrix  (n_items × D)
        print("[ModelStore] Pre-computing item embeddings …")
        self._item_vecs, self._item_ids = self._build_item_index()

        # Build a quick item_id → row-index map for the item matrix
        self._item_id_to_row = {item_id: i for i, item_id in enumerate(self._item_ids)}

        # Metadata
        self.model_version = ckpt_path.name
        self.n_users = cfg.n_users
        self.n_items = cfg.n_items
        self._ready = True
        print(f"[ModelStore] Ready. {self.n_users} users | {self.n_items} items | dim={cfg.output_dim}")

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    # Item index
    # ------------------------------------------------------------------

    def _build_item_index(self) -> tuple[torch.Tensor, np.ndarray]:
        """
        Encode all items in a single forward pass.

        Returns:
            item_vecs : (n_items, D)  L2-normalised, on CPU
            item_ids  : (n_items,)   original movie IDs, matching row order
        """
        movies = self._ds["movies"]
        item_idx   = torch.tensor(movies["item_idx"].values,  dtype=torch.long)
        genre_vecs = torch.tensor(
            np.stack(movies["genre_vec"].values), dtype=torch.float32
        )
        year_norms = torch.tensor(movies["year_norm"].values, dtype=torch.float32)

        with torch.no_grad():
            vecs = self._model.item_tower(item_idx, genre_vecs, year_norms)  # (N, D)

        return vecs.cpu(), movies["item_idx"].values  # return np array of real movie IDs

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _encode_user_by_id(self, user_id: int) -> torch.Tensor:
        """Return (1, D) user vector for a known user_id."""
        user_idx = self._user2idx[user_id]

        # Pull user features from dataset
        users_df = self._ds["users"].set_index("user_idx")
        if user_idx in users_df.index:
            row = users_df.loc[user_idx]
            age_norm = float(row["age_norm"])
            gender_m = float(row["gender_m"])
            occ_idx  = int(row["occ_idx"])
        else:
            age_norm, gender_m, occ_idx = 0.5, 0.5, 12  # defaults

        batch = {
            "user_idx": torch.tensor([user_idx], dtype=torch.long),
            "occ_idx":  torch.tensor([occ_idx],  dtype=torch.long),
            "age_norm": torch.tensor([age_norm], dtype=torch.float32),
            "gender_m": torch.tensor([gender_m], dtype=torch.float32),
        }
        with torch.no_grad():
            return self._model.user_tower(
                batch["user_idx"], batch["occ_idx"],
                batch["age_norm"], batch["gender_m"],
            )  # (1, D)

    def _encode_cold_start(self, liked_movie_ids: list[int]) -> torch.Tensor:
        """
        Approximate user vector by averaging the item vectors of liked movies.
        Then L2-normalise the result.  Simple but surprisingly effective.
        """
        rows = [
            self._item_id_to_row[mid]
            for mid in liked_movie_ids
            if mid in self._item_id_to_row
        ]
        if not rows:
            # Fallback: return a zero vector (will give random-ish scores)
            D = self._item_vecs.shape[1]
            return torch.zeros(1, D)

        vecs = self._item_vecs[rows]               # (k, D)
        avg  = vecs.mean(dim=0, keepdim=True)      # (1, D)
        return torch.nn.functional.normalize(avg, p=2, dim=1)

    def _scores_for_user_vec(
        self,
        user_vec: torch.Tensor,               # (1, D)
        exclude_item_ids: set[int] | None,
        top_k: int,
    ) -> list[ScoredItem]:
        """Dot-product retrieval against the pre-built item index."""
        scores = (user_vec @ self._item_vecs.T).squeeze(0).numpy()  # (n_items,)

        # Build exclusion mask
        if exclude_item_ids:
            for mid in exclude_item_ids:
                row = self._item_id_to_row.get(mid)
                if row is not None:
                    scores[row] = -1.0

        # Top-K by score
        top_rows = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top_rows = top_rows[np.argsort(-scores[top_rows])]

        return [self._row_to_scored_item(row, scores[row]) for row in top_rows]

    def _row_to_scored_item(self, row_idx: int, score: float) -> ScoredItem:
        """Convert an item-matrix row index to a ScoredItem."""
        movie_id = int(self._item_ids[row_idx])
        item_idx = self._item2idx.get(movie_id, -1)

        if item_idx in self._movies_df.index:
            row   = self._movies_df.loc[item_idx]
            title = str(row["title"])
            year  = int(row["year"])
            gvec  = row["genre_vec"]
            genres = [GENRES[i] for i, v in enumerate(gvec) if v > 0]
            poster_url = str(row["poster_url"]) if "poster_url" in row and pd.notna(row.get("poster_url")) else None
            overview   = str(row["overview"])   if "overview"   in row and pd.notna(row.get("overview"))   else None
        else:
            title, year, genres, poster_url, overview = f"Movie {movie_id}", 0, [], None, None

        # Cosine sim is in [-1, 1]; shift to [0, 1] for the API consumer
        normalised_score = float((score + 1.0) / 2.0)
        return ScoredItem(movie_id, title, year, genres, normalised_score, poster_url, overview)

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id:         int | None,
        liked_movie_ids: list[int],
        top_k:           int  = 10,
        exclude_seen:    bool = True,
    ) -> tuple[list[ScoredItem], str]:
        """
        Main recommendation entry point.

        Returns (results, mode) where mode ∈ {"known_user", "cold_start"}.
        """
        if not self._ready:
            raise RuntimeError("ModelStore not loaded")

        # Determine which items to exclude
        exclude: set[int] = set()
        if exclude_seen and user_id is not None and user_id in self._user2idx:
            # Gather all items this user rated in the training set
            u_idx = self._user2idx[user_id]
            rated = self._ds["train"][self._ds["train"]["user_idx"] == u_idx]["item_idx"]
            exclude = {int(self._idx2item[i]) for i in rated if i in self._idx2item}

        # Encode
        if user_id is not None and user_id in self._user2idx:
            user_vec = self._encode_user_by_id(user_id)
            mode = "known_user"
        elif liked_movie_ids:
            user_vec = self._encode_cold_start(liked_movie_ids)
            mode = "cold_start"
        else:
            raise ValueError("Cannot encode user: unknown user_id and no liked_movie_ids")

        results = self._scores_for_user_vec(user_vec, exclude, top_k)
        return results, mode

    def similar(
        self,
        movie_id: int,
        top_k:    int = 10,
    ) -> list[ScoredItem]:
        """Item-to-item similarity: movies closest to the seed in embedding space."""
        if not self._ready:
            raise RuntimeError("ModelStore not loaded")

        row = self._item_id_to_row.get(movie_id)
        if row is None:
            raise KeyError(f"movie_id {movie_id} not in index")

        seed_vec = self._item_vecs[row].unsqueeze(0)  # (1, D)
        # Exclude the seed itself
        return self._scores_for_user_vec(seed_vec, exclude_item_ids={movie_id}, top_k=top_k)

    def get_movie(self, movie_id: int) -> ScoredItem | None:
        """Look up a single movie by ID (score field will be 0.0)."""
        row = self._item_id_to_row.get(movie_id)
        if row is None:
            return None
        return self._row_to_scored_item(row, 0.0)

    def known_user_ids(self) -> list[int]:
        return sorted(self._user2idx.keys())


# ---------------------------------------------------------------------------
# Module-level singleton — imported by main.py
# ---------------------------------------------------------------------------

store = ModelStore()