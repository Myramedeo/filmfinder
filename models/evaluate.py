"""
evaluate.py
-----------
Offline ranking metrics for the two-tower recommender.

Metrics implemented
-------------------
  - AUC-ROC          (interaction-level, fast proxy during training)
  - Precision@K      (user-level, averaged)
  - Recall@K         (user-level, averaged)
  - NDCG@K           (user-level, averaged) 

Evaluation protocol
-------------------
For each user in the evaluation set:
  1. Collect all items the user interacted with positively (label = 1).
  2. Score every item the user has NOT seen in training using the model.
  3. Rank all candidate items by predicted score descending.
  4. Compute precision / recall / NDCG within the top-K.

This mirrors the real evaluation used in industry (leave-one-out or
leave-last-N-out).  In practice we use the full test set so the metric
is stable.
"""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Interaction-level metrics  (fast, used every epoch)
# ---------------------------------------------------------------------------

def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Approximate AUC via the Wilcoxon-Mann-Whitney statistic."""
    pos_mask = labels == 1
    neg_mask = labels == 0
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return float("nan")
    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]
    # Vectorised: count (pos > neg) + 0.5 * (pos == neg)
    diff = pos_scores[:, None] - neg_scores[None, :]
    auc = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(auc) / (pos_mask.sum() * neg_mask.sum())


# ---------------------------------------------------------------------------
# Ranking metrics helpers
# ---------------------------------------------------------------------------

def _dcg(relevances: list[int], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    return sum(
        rel / math.log2(rank + 2)
        for rank, rel in enumerate(relevances[:k])
    )


def _ndcg_at_k(relevances: list[int], k: int) -> float:
    """Normalised DCG at K."""
    ideal = sorted(relevances, reverse=True)
    idcg  = _dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return _dcg(relevances, k) / idcg


def _precision_at_k(relevances: list[int], k: int) -> float:
    return sum(relevances[:k]) / k


def _recall_at_k(relevances: list[int], k: int, n_pos: int) -> float:
    if n_pos == 0:
        return 0.0
    return sum(relevances[:k]) / n_pos


# ---------------------------------------------------------------------------
# Per-user ranking evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def ranking_metrics(
    model: torch.nn.Module,
    test_interactions: "pd.DataFrame",
    train_interactions: "pd.DataFrame",
    item_features: "pd.DataFrame",
    device: torch.device,
    k_values: list[int] = [5, 10, 20],
    max_users: int | None = 200,   # cap for speed; set None for full eval
) -> dict[str, float]:
    """
    Compute user-level Precision@K, Recall@K, NDCG@K.

    For each test user:
      - Positive items  = items rated ≥ threshold in test set
      - Candidate items = all items NOT seen in training (to avoid trivial recall)
      - Score each candidate with the model, rank, evaluate top-K

    Args:
        model:              trained TwoTowerModel
        test_interactions:  DataFrame with columns [user_idx, item_idx, label]
        train_interactions: DataFrame with columns [user_idx, item_idx, label]
        item_features:      movies DataFrame from build_dataset()
        device:             torch device
        k_values:           list of K values to report
        max_users:          evaluate at most this many users (for speed)
    """
    model.eval()

    # Build per-user look-ups
    train_seen: dict[int, set[int]] = defaultdict(set)
    for row in train_interactions.itertuples():
        train_seen[row.user_idx].add(row.item_idx)

    test_pos: dict[int, set[int]] = defaultdict(set)
    for row in test_interactions.itertuples():
        if row.label == 1.0:
            test_pos[row.user_idx].add(row.item_idx)

    # Pre-compute ALL item embeddings once (cheap: ~1682 items)
    all_item_idx   = torch.tensor(item_features["item_idx"].values,  dtype=torch.long).to(device)
    all_genre_vec  = torch.tensor(
        np.stack(item_features["genre_vec"].values), dtype=torch.float32
    ).to(device)
    all_year_norm  = torch.tensor(item_features["year_norm"].values, dtype=torch.float32).to(device)

    item_vecs = model.item_tower(all_item_idx, all_genre_vec, all_year_norm)  # (n_items, D)
    item_idx_list = item_features["item_idx"].values                          # np array

    # Collect user features for test users
    test_users = list(test_pos.keys())
    if max_users is not None:
        test_users = test_users[:max_users]

    results: dict[str, list[float]] = {
        f"P@{k}": [] for k in k_values
    } | {
        f"R@{k}": [] for k in k_values
    } | {
        f"NDCG@{k}": [] for k in k_values
    }

    # We need user features; pull from test_interactions
    user_feat_df = test_interactions.drop_duplicates("user_idx").set_index("user_idx")

    for user_idx in test_users:
        if user_idx not in test_pos or len(test_pos[user_idx]) == 0:
            continue

        # Build user batch (single user, repeated for vectorised tower call)
        row = user_feat_df.loc[user_idx] if user_idx in user_feat_df.index else None
        if row is None:
            continue

        user_batch = {
            "user_idx": torch.tensor([user_idx],          dtype=torch.long).to(device),
            "occ_idx":  torch.tensor([int(row.get("occ_idx", 0))], dtype=torch.long).to(device),
            "age_norm": torch.tensor([float(row.get("age_norm", 0.5))], dtype=torch.float32).to(device),
            "gender_m": torch.tensor([float(row.get("gender_m", 0.5))], dtype=torch.float32).to(device),
        }
        user_vec = model.user_tower(
            user_batch["user_idx"],
            user_batch["occ_idx"],
            user_batch["age_norm"],
            user_batch["gender_m"],
        )  # (1, D)

        # Cosine similarity against all items
        scores = (user_vec @ item_vecs.T).squeeze(0)          # (n_items,)
        scores_np = scores.cpu().numpy()

        # Filter out training items
        seen = train_seen.get(user_idx, set())
        mask = np.array([idx not in seen for idx in item_idx_list], dtype=bool)
        candidate_items = item_idx_list[mask]
        candidate_scores = scores_np[mask]

        # Sort by score descending
        order = np.argsort(-candidate_scores)
        ranked_items = candidate_items[order]

        # Ground truth
        pos_items = test_pos[user_idx]
        relevances = [1 if item in pos_items else 0 for item in ranked_items]
        n_pos = len(pos_items)

        for k in k_values:
            results[f"P@{k}"].append(_precision_at_k(relevances, k))
            results[f"R@{k}"].append(_recall_at_k(relevances, k, n_pos))
            results[f"NDCG@{k}"].append(_ndcg_at_k(relevances, k))

    return {metric: float(np.mean(vals)) for metric, vals in results.items() if vals}


# ---------------------------------------------------------------------------
# Batch-level AUC helper (called every epoch from trainer)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_auc(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    all_labels, all_scores = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = model(batch)
        all_labels.append(batch["label"].cpu().numpy())
        all_scores.append(preds.cpu().numpy())
    return binary_auc(
        np.concatenate(all_labels),
        np.concatenate(all_scores),
    )