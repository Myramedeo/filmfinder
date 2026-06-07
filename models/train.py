"""
train.py
--------
Training loop for the two-tower recommender.

Features
--------
  - Binary cross-entropy loss (positive / negative interactions)
  - AdamW optimizer with weight decay
  - CosineAnnealingLR scheduler
  - Per-epoch validation AUC with early stopping
  - Checkpoint saving (best val AUC)
  - Full ranking evaluation (NDCG@K, P@K, R@K) after training
  - CPU + GPU (CUDA / MPS) support

Usage
-----
    # from project root
    python -m models.train

    # override hyper-params
    python -m models.train --epochs 20 --lr 3e-4 --batch-size 2048 --output-dim 128
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.preprocess import build_dataset
from data.dataset import make_loaders
from models.two_tower import TwoTowerModel, build_model
from models.evaluate import compute_auc, ranking_metrics

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def bce_loss(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with label smoothing to reduce overconfidence."""
    smooth = 0.05
    labels_smooth = labels * (1 - smooth) + 0.5 * smooth
    return nn.functional.binary_cross_entropy(preds, labels_smooth)


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: TwoTowerModel,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        preds = model(batch)
        loss  = bce_loss(preds, batch["label"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(batch["label"])
    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

def train(
    epochs:      int   = 15,
    lr:          float = 1e-3,
    weight_decay:float = 1e-4,
    batch_size:  int   = 1024,
    output_dim:  int   = 64,
    dropout:     float = 0.2,
    patience:    int   = 4,
    k_values:    list  = [5, 10, 20],
    device_name: str   = "auto",
    save_dir:    Path  = CHECKPOINT_DIR,
) -> dict:
    """
    Full training run. Returns a results dict with final metrics.
    """
    # ── Device ────────────────────────────────────────────────────────────
    if device_name == "auto":
        device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps")  if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device(device_name)
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\nBuilding dataset …")
    ds = build_dataset()
    train_loader, val_loader, test_loader = make_loaders(ds, batch_size=batch_size)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(ds, output_dim=output_dim, dropout=dropout).to(device)
    counts = model.param_count()
    print(f"\nModel: {counts['total']:,} parameters  "
          f"(user tower {counts['user_tower']:,}  |  item tower {counts['item_tower']:,})")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # ── Training loop ─────────────────────────────────────────────────────
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_model.pt"

    best_val_auc = 0.0
    patience_counter = 0
    history: list[dict] = []

    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Val AUC':>9}  {'Time':>7}  {'LR':>10}")
    print("─" * 56)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_auc    = compute_auc(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        marker = "  ◀ best" if val_auc > best_val_auc else ""
        print(f"{epoch:>6}  {train_loss:>11.5f}  {val_auc:>9.4f}  {elapsed:>6.1f}s  {lr_now:>10.2e}{marker}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc})

        # ── Checkpoint best model ─────────────────────────────────────────
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(
                {
                    "epoch":        epoch,
                    "model_state":  model.state_dict(),
                    "cfg":          model.cfg,
                    "val_auc":      val_auc,
                    "user2idx":     ds["user2idx"],
                    "item2idx":     ds["item2idx"],
                    "idx2item":     ds["idx2item"],
                },
                best_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    print(f"\nBest val AUC: {best_val_auc:.4f}  (saved to {best_path})")

    # ── Load best weights for ranking eval ────────────────────────────────
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    print("\nRunning ranking evaluation on test set …")
    ranking = ranking_metrics(
        model=model,
        test_interactions=ds["test"],
        train_interactions=ds["train"],
        item_features=ds["movies"],
        device=device,
        k_values=k_values,
        max_users=None,  # set to e.g. 200 for a quick smoke test; None for full eval
    )

    test_auc = compute_auc(model, test_loader, device)

    print(f"\n{'─'*40}")
    print(f"  Test AUC   : {test_auc:.4f}")
    for metric, val in sorted(ranking.items()):
        print(f"  {metric:<10} : {val:.4f}")
    print(f"{'─'*40}")

    # Save results JSON alongside checkpoint
    results = {"test_auc": test_auc, **ranking, "history": history}
    results_path = save_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train two-tower recommender")
    parser.add_argument("--epochs",      type=int,   default=15)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch-size",  type=int,   default=1024)
    parser.add_argument("--output-dim",  type=int,   default=64)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--patience",    type=int,   default=4)
    parser.add_argument("--device",      type=str,   default="auto")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        output_dim=args.output_dim,
        dropout=args.dropout,
        patience=args.patience,
        device_name=args.device,
    )