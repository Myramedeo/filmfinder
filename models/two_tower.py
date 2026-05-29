"""
two_tower.py
------------
Two-tower (dual-encoder) neural network for movie recommendation.

Design choices
--------------
- Separate embedding tables for users and items (standard two-tower).
- Occupation gets its own small embedding so the model learns job-group similarity.
- Genre vector goes through a lightweight MLP rather than a raw linear layer,
  so the model can learn non-linear genre combinations.
- Final towers produce L2-normalised vectors; scoring is cosine similarity
  scaled to (0,1) via sigmoid.  This keeps the dot product bounded and
  makes the model compatible with ANN retrieval at inference time.
- Dropout applied in tower MLPs for regularisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Config dataclass — keeps model hyper-params in one place
# ---------------------------------------------------------------------------

@dataclass
class TwoTowerConfig:
    # Vocabulary sizes (set from build_dataset stats)
    n_users:       int = 943
    n_items:       int = 1_682
    n_occupations: int = 21     # MovieLens 100K occupation categories
    n_genres:      int = 19     # MovieLens 100K genre columns

    # Embedding dims
    user_emb_dim:  int = 64
    item_emb_dim:  int = 64
    occ_emb_dim:   int = 16
    genre_hidden:  int = 32     # hidden size of the genre MLP

    # Tower MLP hidden sizes  (excluding input / output)
    tower_hidden:  int = 128
    output_dim:    int = 64     # shared output dimension D

    # Regularisation
    dropout:       float = 0.2

    # Scoring
    temperature:   float = 1.0  # scales logit before sigmoid; learnable or fixed


# ---------------------------------------------------------------------------
# Reusable MLP block
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Linear → BN → ReLU → Dropout, repeated `n_layers` times."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:          # no BN/activation on last layer
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# User tower
# ---------------------------------------------------------------------------

class UserTower(nn.Module):
    """
    Encodes a user into a dense vector.

    Inputs (per sample):
        user_idx  : (B,)       int64
        occ_idx   : (B,)       int64
        age_norm  : (B,)       float32
        gender_m  : (B,)       float32

    Output:
        user_vec  : (B, output_dim)  float32, L2-normalised
    """

    def __init__(self, cfg: TwoTowerConfig) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(cfg.n_users,       cfg.user_emb_dim)
        self.occ_emb  = nn.Embedding(cfg.n_occupations, cfg.occ_emb_dim)

        in_dim = cfg.user_emb_dim + cfg.occ_emb_dim + 2  # +2 for age, gender
        self.mlp = MLP(
            in_dim=in_dim,
            hidden_dim=cfg.tower_hidden,
            out_dim=cfg.output_dim,
            n_layers=2,
            dropout=cfg.dropout,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.occ_emb.weight,  std=0.01)

    def forward(
        self,
        user_idx: torch.Tensor,
        occ_idx:  torch.Tensor,
        age_norm: torch.Tensor,
        gender_m: torch.Tensor,
    ) -> torch.Tensor:
        u = self.user_emb(user_idx)                     # (B, user_emb_dim)
        o = self.occ_emb(occ_idx)                       # (B, occ_emb_dim)
        cont = torch.stack([age_norm, gender_m], dim=1) # (B, 2)
        x = torch.cat([u, o, cont], dim=1)              # (B, in_dim)
        out = self.mlp(x)                               # (B, output_dim)
        return F.normalize(out, p=2, dim=1)             # L2 normalise


# ---------------------------------------------------------------------------
# Item tower
# ---------------------------------------------------------------------------

class ItemTower(nn.Module):
    """
    Encodes a movie into a dense vector.

    Inputs (per sample):
        item_idx  : (B,)       int64
        genre_vec : (B, 19)    float32  multi-hot
        year_norm : (B,)       float32

    Output:
        item_vec  : (B, output_dim)  float32, L2-normalised
    """

    def __init__(self, cfg: TwoTowerConfig) -> None:
        super().__init__()
        self.item_emb  = nn.Embedding(cfg.n_items, cfg.item_emb_dim)
        self.genre_mlp = MLP(
            in_dim=cfg.n_genres,
            hidden_dim=cfg.genre_hidden,
            out_dim=cfg.genre_hidden,
            n_layers=2,
            dropout=cfg.dropout,
        )
        in_dim = cfg.item_emb_dim + cfg.genre_hidden + 1  # +1 for year
        self.mlp = MLP(
            in_dim=in_dim,
            hidden_dim=cfg.tower_hidden,
            out_dim=cfg.output_dim,
            n_layers=2,
            dropout=cfg.dropout,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(
        self,
        item_idx:  torch.Tensor,
        genre_vec: torch.Tensor,
        year_norm: torch.Tensor,
    ) -> torch.Tensor:
        m  = self.item_emb(item_idx)           # (B, item_emb_dim)
        g  = self.genre_mlp(genre_vec)         # (B, genre_hidden)
        yr = year_norm.unsqueeze(1)            # (B, 1)
        x  = torch.cat([m, g, yr], dim=1)     # (B, in_dim)
        out = self.mlp(x)                      # (B, output_dim)
        return F.normalize(out, p=2, dim=1)    # L2 normalise


# ---------------------------------------------------------------------------
# Full two-tower model
# ---------------------------------------------------------------------------

class TwoTowerModel(nn.Module):
    """
    Combines UserTower + ItemTower.  Forward pass returns a scalar logit in
    (0, 1) for each (user, item) pair via cosine similarity → sigmoid.

    The model is also used at inference time as an encoder-only system:
      - encode_user()  →  user_vec  for ANN indexing
      - encode_items() →  item_vecs for building the retrieval index
    """

    def __init__(self, cfg: TwoTowerConfig) -> None:
        super().__init__()
        self.cfg        = cfg
        self.user_tower = UserTower(cfg)
        self.item_tower = ItemTower(cfg)
        # Learnable temperature parameter (log-scale for numerical stability)
        self.log_temp = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(min=0.01, max=10.0)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: dict from MovieLensDataset.__getitem__ (see dataset.py)
        Returns:
            logits: (B,) float32 — predicted probability of interaction
        """
        user_vec = self.user_tower(
            batch["user_idx"],
            batch["occ_idx"],
            batch["age_norm"],
            batch["gender_m"],
        )
        item_vec = self.item_tower(
            batch["item_idx"],
            batch["genre_vec"],
            batch["year_norm"],
        )
        # Cosine similarity is already bounded [-1, 1] because both vecs are L2-normalised
        cos_sim = (user_vec * item_vec).sum(dim=1)      # (B,)
        logits  = cos_sim / self.temperature            # scale
        return torch.sigmoid(logits)                    # → (0, 1)

    # ------------------------------------------------------------------
    # Encoder helpers for inference / retrieval index building
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_user(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return L2-normalised user vector. Shape: (B, output_dim)."""
        self.eval()
        return self.user_tower(
            batch["user_idx"],
            batch["occ_idx"],
            batch["age_norm"],
            batch["gender_m"],
        )

    @torch.no_grad()
    def encode_items(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return L2-normalised item vectors. Shape: (B, output_dim)."""
        self.eval()
        return self.item_tower(
            batch["item_idx"],
            batch["genre_vec"],
            batch["year_norm"],
        )

    # ------------------------------------------------------------------
    # Convenience: parameter count
    # ------------------------------------------------------------------

    def param_count(self) -> dict[str, int]:
        user_p = sum(p.numel() for p in self.user_tower.parameters())
        item_p = sum(p.numel() for p in self.item_tower.parameters())
        total  = sum(p.numel() for p in self.parameters())
        return {"user_tower": user_p, "item_tower": item_p, "total": total}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(dataset_dict: dict, **overrides) -> TwoTowerModel:
    """
    Construct a TwoTowerModel sized to match the dataset.

    Args:
        dataset_dict: returned by preprocess.build_dataset()
        **overrides:  any TwoTowerConfig field to change, e.g. output_dim=128
    """
    cfg = TwoTowerConfig(
        n_users=dataset_dict["n_users"],
        n_items=dataset_dict["n_items"],
        **overrides,
    )
    return TwoTowerModel(cfg)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.preprocess import build_dataset
    from data.dataset import make_loaders

    ds = build_dataset()
    train_loader, *_ = make_loaders(ds, batch_size=64)
    batch = next(iter(train_loader))

    model = build_model(ds)
    counts = model.param_count()
    print(f"\nModel parameter counts:")
    print(f"  user tower : {counts['user_tower']:,}")
    print(f"  item tower : {counts['item_tower']:,}")
    print(f"  total      : {counts['total']:,}")

    preds = model(batch)
    print(f"\nForward pass OK — output shape: {tuple(preds.shape)}, range: [{preds.min():.3f}, {preds.max():.3f}]")

    user_vecs = model.encode_user(batch)
    item_vecs = model.encode_items(batch)
    print(f"encode_user : {tuple(user_vecs.shape)}  norm={user_vecs.norm(dim=1).mean():.4f}")
    print(f"encode_items: {tuple(item_vecs.shape)}  norm={item_vecs.norm(dim=1).mean():.4f}")
    print("\nTemperature (learnable):", model.temperature.item())