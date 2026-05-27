"""
dataset.py
----------
PyTorch Dataset and DataLoader factories for the two-tower recommender.

  User tower inputs:
    user_idx    (int64)   — index into the user embedding table
    age_norm    (float32) — normalised age  [0, 1]
    gender_m    (float32) — 1 = male, 0 = female
    occ_idx     (int64)   — occupation class index

  Item tower inputs:
    item_idx    (int64)   — index into the item embedding table
    genre_vec   (float32, 19) — multi-hot genre vector
    year_norm   (float32) — normalised release year [0, 1]

  Label:
    label       (float32) — 1/0 (binary) or rating/5 (regression)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class MovieLensDataset(Dataset):
    """
    Efficient dataset that avoids per-sample DataFrame lookups by pre-converting
    all columns to numpy arrays at construction time.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        movies: pd.DataFrame,
        users: pd.DataFrame,
    ) -> None:
        # ------------------------------------------------------------------
        # Build O(1) look-up arrays indexed by contiguous idx
        # ------------------------------------------------------------------
        n_items = movies["item_idx"].max() + 1
        n_users = users["user_idx"].max() + 1

        # Item feature arrays  (shape: n_items × *)
        genre_mat   = np.zeros((n_items, 19), dtype=np.float32)
        year_arr    = np.zeros(n_items, dtype=np.float32)

        for _, row in movies.iterrows():
            idx = int(row["item_idx"])
            genre_mat[idx] = np.array(row["genre_vec"], dtype=np.float32)
            year_arr[idx]  = float(row["year_norm"])

        self._genre_mat = genre_mat
        self._year_arr  = year_arr

        # User feature arrays  (shape: n_users × *)
        age_arr    = np.zeros(n_users, dtype=np.float32)
        gender_arr = np.zeros(n_users, dtype=np.float32)
        occ_arr    = np.zeros(n_users, dtype=np.int64)

        for _, row in users.iterrows():
            idx = int(row["user_idx"])
            age_arr[idx]    = float(row["age_norm"])
            gender_arr[idx] = float(row["gender_m"])
            occ_arr[idx]    = int(row["occ_idx"])

        self._age_arr    = age_arr
        self._gender_arr = gender_arr
        self._occ_arr    = occ_arr

        # ------------------------------------------------------------------
        # Interaction columns → numpy for fast indexing
        # ------------------------------------------------------------------
        self._user_idx = interactions["user_idx"].values.astype(np.int64)
        self._item_idx = interactions["item_idx"].values.astype(np.int64)
        self._labels   = interactions["label"].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        u = self._user_idx[idx]
        m = self._item_idx[idx]

        return {
            # User tower
            "user_idx":  torch.tensor(u, dtype=torch.long),
            "age_norm":  torch.tensor(self._age_arr[u],    dtype=torch.float32),
            "gender_m":  torch.tensor(self._gender_arr[u], dtype=torch.float32),
            "occ_idx":   torch.tensor(self._occ_arr[u],    dtype=torch.long),
            # Item tower
            "item_idx":  torch.tensor(m, dtype=torch.long),
            "genre_vec": torch.tensor(self._genre_mat[m],  dtype=torch.float32),
            "year_norm": torch.tensor(self._year_arr[m],   dtype=torch.float32),
            # Supervision
            "label":     torch.tensor(self._labels[idx],   dtype=torch.float32),
        }


def make_loaders(
    dataset_dict: dict,
    batch_size: int = 1024,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience factory.  Pass the dict returned by preprocess.build_dataset().

    Returns (train_loader, val_loader, test_loader).
    """
    movies = dataset_dict["movies"]
    users  = dataset_dict["users"]

    train_ds = MovieLensDataset(dataset_dict["train"], movies, users)
    val_ds   = MovieLensDataset(dataset_dict["val"],   movies, users)
    test_ds  = MovieLensDataset(dataset_dict["test"],  movies, users)

    shared = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)

    train_loader = DataLoader(train_ds, shuffle=True,  **shared)
    val_loader   = DataLoader(val_ds,   shuffle=False, **shared)
    test_loader  = DataLoader(test_ds,  shuffle=False, **shared)

    print(
        f"DataLoaders ready — "
        f"train {len(train_ds):,}  |  val {len(val_ds):,}  |  test {len(test_ds):,}  "
        f"(batch_size={batch_size})"
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from download import download_movielens
    from preprocess import build_dataset

    download_movielens()
    ds = build_dataset()
    train_loader, val_loader, test_loader = make_loaders(ds, batch_size=4)

    batch = next(iter(train_loader))
    print("\nSample batch keys:", list(batch.keys()))
    for k, v in batch.items():
        print(f"  {k:<12} {tuple(v.shape)}  dtype={v.dtype}")