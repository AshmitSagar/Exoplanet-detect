"""
checkpoint.py — Checkpoint save / load and best-model tracking.

Design notes
────────────
• A checkpoint is a single dict saved with torch.save.  This makes it trivial
  to inspect with torch.load and to extend with new fields (e.g., scaler state
  for AMP) without breaking old checkpoints.

• CheckpointManager encapsulates the logic for deciding when to write
  "best_model.pth" vs "last_model.pth" so that trainer.py has no conditional
  bookkeeping.

• load_checkpoint is side-effect-free with respect to the file system — it
  only modifies model/optimizer/scheduler in-place.

Checkpoint dict schema
──────────────────────
{
    "epoch"         : int,
    "model_state"   : OrderedDict,
    "optim_state"   : dict,
    "sched_state"   : dict | None,
    "scaler_state"  : dict | None,   # AMP GradScaler
    "best_val_loss" : float,
    "config"        : dict,          # TrainingConfig.__dict__ (for reference only)
    "metrics"       : dict,          # metrics at this checkpoint
}
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    """
    Atomically save a checkpoint dict to `path`.

    The write goes to a temporary file first, then is renamed to `path`
    so a partial write never corrupts the checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    shutil.move(str(tmp), str(path))
    log.debug("Checkpoint written → %s", path)


def load_checkpoint(
    path      : Path,
    model     : nn.Module,
    optimizer : Optional[Optimizer]  = None,
    scheduler : Optional[_LRScheduler] = None,
    scaler    : Optional[Any]        = None,   # torch.cuda.amp.GradScaler
    device    : Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint from `path` and restore state into the provided objects.

    All arguments after `path` are optional.  Only the objects that are
    provided (non-None) will have their state restored.

    Returns:
        The full checkpoint dict so the caller can read "epoch",
        "best_val_loss", "metrics", etc.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device if device is not None else "cpu"
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])

    if scheduler is not None and ckpt.get("sched_state") is not None:
        scheduler.load_state_dict(ckpt["sched_state"])

    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    log.info(
        "Resumed from %s  (epoch %d, best_val_loss=%.4f)",
        path, ckpt.get("epoch", 0), ckpt.get("best_val_loss", float("inf"))
    )
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# High-level manager
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Manages checkpoint saving decisions during a training run.

    Responsibilities
    ────────────────
    1. Track the best validation loss seen so far.
    2. Save "best_model.pth"  whenever val_loss improves.
    3. Save "last_model.pth"  at every call to `step()` (overwriting).
    4. Save a numbered snapshot "epoch_{N:04d}.pth" every `save_every` epochs
       (configurable; 0 to disable numbered snapshots).

    Args:
        checkpoint_dir : directory where .pth files are written
        save_every     : save a numbered snapshot every N epochs (0 = disabled)
        min_delta      : minimum improvement in val_loss to count as new best

    Example
    ───────
        manager = CheckpointManager(cfg.checkpoint_dir, save_every=cfg.save_every)
        for epoch in range(num_epochs):
            ...
            state = build_state(model, optimizer, scheduler, scaler, epoch, metrics)
            is_best = manager.step(val_loss, state, epoch)
    """

    def __init__(
        self,
        checkpoint_dir : Path,
        save_every     : int   = 5,
        min_delta      : float = 1e-4,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every     = save_every
        self.min_delta      = min_delta
        self.best_val_loss  : float = float("inf")
        self.best_epoch     : int   = -1

    @property
    def best_model_path(self) -> Path:
        return self.checkpoint_dir / "best_model.pth"

    @property
    def last_model_path(self) -> Path:
        return self.checkpoint_dir / "last_model.pth"

    def step(
        self,
        val_loss : float,
        state    : Dict[str, Any],
        epoch    : int,
    ) -> bool:
        """
        Evaluate val_loss, save appropriate checkpoints, return True if new best.

        Always saves "last_model.pth".
        Saves "best_model.pth" if val_loss improved by at least `min_delta`.
        Saves "epoch_{N:04d}.pth" if save_every > 0 and epoch is a multiple.
        """
        # Always persist the latest model
        save_checkpoint(state, self.last_model_path)

        # Periodic numbered snapshot
        if self.save_every > 0 and (epoch % self.save_every == 0):
            snap_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pth"
            save_checkpoint(state, snap_path)
            log.info("Snapshot → %s", snap_path)

        # Best-model check
        is_best = val_loss < self.best_val_loss - self.min_delta
        if is_best:
            self.best_val_loss = val_loss
            self.best_epoch    = epoch
            save_checkpoint(state, self.best_model_path)
            log.info(
                "New best model at epoch %d  val_loss=%.4f → %s",
                epoch, val_loss, self.best_model_path
            )

        return is_best

    def has_improved(self, val_loss: float) -> bool:
        """Return True if val_loss is a new best (without saving)."""
        return val_loss < self.best_val_loss - self.min_delta
