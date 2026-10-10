"""
trainer.py — Trainer: orchestrates the epoch loop, AMP, early stopping, and logging.

Design notes
────────────
• Trainer does NOT own the model, dataloaders, or optimizer — those are
  constructed in train.py and injected.  This keeps Trainer testable and
  reusable without importing the whole training stack.

• Mixed precision (AMP) uses torch.cuda.amp.GradScaler + autocast.  If CUDA
  is absent the scaler is a no-op stub so the code path is identical.

• CSV logging writes one row per epoch to `<log_dir>/train_log.csv`.
  Columns: epoch, train_loss, train_acc, val_loss, val_acc, val_f1_macro,
           val_precision_macro, val_recall_macro, lr.

• History plots (loss + accuracy) are saved to <plot_dir>/ at run-end.
  They are deliberately not shown interactively (plt.show is never called)
  so the code runs headlessly on a server.

• Early stopping is implemented via a simple counter that increments when
  the validation loss fails to improve and resets on improvement.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import CheckpointManager, save_checkpoint
from .config     import TrainingConfig
from .metrics    import MetricAccumulator

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# AMP compatibility shim
# ─────────────────────────────────────────────────────────────────────────────

def _make_scaler(enabled: bool):
    """Return a GradScaler if AMP is enabled and CUDA is present, else a no-op."""
    if enabled and torch.cuda.is_available():
        return torch.amp.GradScaler("cuda")
    # No-op fallback: an object whose methods are identity functions
    class _NoOpScaler:
        def scale(self, loss):   return loss
        def step(self, opt):     opt.step()
        def update(self):        pass
        def state_dict(self):    return {}
        def load_state_dict(self, sd): pass
    return _NoOpScaler()


def _autocast(enabled: bool, device: torch.device):
    """Return the appropriate autocast context manager."""
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda")
    # CPU fallback — autocast with cpu device type (no-op in practice)
    import contextlib
    return contextlib.nullcontext()


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Orchestrates training and validation loops for AstroNetCNN.

    Args:
        model        : AstroNetCNN instance (already on device).
        config       : TrainingConfig.
        train_loader : DataLoader for the training split.
        val_loader   : DataLoader for the validation split.
        optimizer    : AdamW (or any Optimizer) bound to model parameters.
        scheduler    : LR scheduler (CosineAnnealingLR or StepLR).
        device       : torch.device ("cuda" or "cpu").
        criterion    : Loss function (typically nn.CrossEntropyLoss with weights).
    """

    def __init__(
        self,
        model        : nn.Module,
        config       : TrainingConfig,
        train_loader : DataLoader,
        val_loader   : DataLoader,
        optimizer    : Optimizer,
        scheduler    : _LRScheduler,
        device       : torch.device,
        criterion    : nn.Module,
    ) -> None:
        self.model        = model
        self.cfg          = config
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.device       = device
        self.criterion    = criterion

        self.scaler = _make_scaler(config.mixed_precision)

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir = config.checkpoint_dir,
            save_every     = config.save_every,
            min_delta      = config.min_delta,
        )

        # History lists populated during fit()
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
            "val_f1":     [], "lr":        [],
        }

        # CSV log file handle (opened lazily in fit())
        self._csv_path: Path = config.log_dir / "train_log.csv"
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_fh    = None

    # ─── Epoch loops ─────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch; return aggregated metrics."""
        self.model.train()
        acc = MetricAccumulator()

        pbar = tqdm(
            self.train_loader,
            desc  = f"  Train {epoch:>4}",
            leave = False,
            unit  = "batch",
            dynamic_ncols = True,
        )

        for batch in pbar:
            global_view = batch["global_view"].to(self.device, non_blocking=True)
            local_view  = batch["local_view"].to(self.device, non_blocking=True)
            labels      = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with _autocast(self.cfg.mixed_precision, self.device):
                logits = self.model(global_view, local_view)   # (B, C)
                loss   = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            # Gradient clipping (unscaled) to prevent exploding gradients
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            preds = logits.detach().argmax(dim=1)
            acc.update(loss.item(), preds, labels)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{(preds == labels).float().mean().item():.3f}",
            )

        return acc.compute(self.cfg.num_classes)

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one validation epoch; return aggregated metrics."""
        self.model.eval()
        acc = MetricAccumulator()

        pbar = tqdm(
            self.val_loader,
            desc  = f"  Val   {epoch:>4}",
            leave = False,
            unit  = "batch",
            dynamic_ncols = True,
        )

        for batch in pbar:
            global_view = batch["global_view"].to(self.device, non_blocking=True)
            local_view  = batch["local_view"].to(self.device, non_blocking=True)
            labels      = batch["label"].to(self.device, non_blocking=True)

            with _autocast(self.cfg.mixed_precision, self.device):
                logits = self.model(global_view, local_view)
                loss   = self.criterion(logits, labels)

            preds = logits.argmax(dim=1)
            acc.update(loss.item(), preds, labels)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{(preds == labels).float().mean().item():.3f}",
            )

        return acc.compute(self.cfg.num_classes)

    # ─── CSV logger ──────────────────────────────────────────────────────────

    def _open_csv(self) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_fh = open(self._csv_path, "w", newline="", encoding="utf-8")
        fieldnames = [
            "epoch",
            "train_loss", "train_acc",
            "val_loss",   "val_acc",
            "val_f1_macro", "val_precision_macro", "val_recall_macro",
            "lr",
        ]
        # Extend with per-class F1 columns
        for i in range(self.cfg.num_classes):
            fieldnames += [f"val_f1_cls_{i}", f"val_precision_cls_{i}", f"val_recall_cls_{i}"]

        self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=fieldnames, extrasaction="ignore")
        self._csv_writer.writeheader()

    def _log_csv(
        self,
        epoch       : int,
        train_m     : Dict[str, float],
        val_m       : Dict[str, float],
        current_lr  : float,
    ) -> None:
        row = {
            "epoch"                : epoch,
            "train_loss"           : f"{train_m['loss']:.6f}",
            "train_acc"            : f"{train_m['accuracy']:.6f}",
            "val_loss"             : f"{val_m['loss']:.6f}",
            "val_acc"              : f"{val_m['accuracy']:.6f}",
            "val_f1_macro"         : f"{val_m['f1_macro']:.6f}",
            "val_precision_macro"  : f"{val_m['precision_macro']:.6f}",
            "val_recall_macro"     : f"{val_m['recall_macro']:.6f}",
            "lr"                   : f"{current_lr:.8f}",
        }
        for i in range(self.cfg.num_classes):
            row[f"val_f1_cls_{i}"]        = f"{val_m.get(f'f1_cls_{i}', 0.0):.6f}"
            row[f"val_precision_cls_{i}"] = f"{val_m.get(f'precision_cls_{i}', 0.0):.6f}"
            row[f"val_recall_cls_{i}"]    = f"{val_m.get(f'recall_cls_{i}', 0.0):.6f}"
        self._csv_writer.writerow(row)
        self._csv_fh.flush()

    # ─── History plots ────────────────────────────────────────────────────────

    def _save_plots(self) -> None:
        """Save loss and accuracy training-history plots to plot_dir."""
        try:
            import matplotlib
            matplotlib.use("Agg")   # headless backend — never calls plt.show
            import matplotlib.pyplot as plt

            epochs = list(range(1, len(self.history["train_loss"]) + 1))

            # ── Loss ──────────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, self.history["train_loss"], label="Train loss", linewidth=1.8)
            ax.plot(epochs, self.history["val_loss"],   label="Val loss",   linewidth=1.8)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cross-entropy loss")
            ax.set_title("AstroNetCNN — Loss")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            loss_path = self.cfg.plot_dir / "loss_curve.png"
            fig.savefig(loss_path, dpi=150)
            plt.close(fig)
            log.info("Loss plot → %s", loss_path)

            # ── Accuracy ──────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, self.history["train_acc"], label="Train acc", linewidth=1.8)
            ax.plot(epochs, self.history["val_acc"],   label="Val acc",   linewidth=1.8)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0.0, 1.05)
            ax.set_title("AstroNetCNN — Accuracy")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            acc_path = self.cfg.plot_dir / "accuracy_curve.png"
            fig.savefig(acc_path, dpi=150)
            plt.close(fig)
            log.info("Accuracy plot → %s", acc_path)

        except Exception as exc:
            log.warning("Could not save plots: %s", exc)

    # ─── Main loop ────────────────────────────────────────────────────────────

    def fit(self, start_epoch: int = 0) -> Dict[str, List[float]]:
        """
        Run the full training loop from `start_epoch` to `cfg.num_epochs`.

        Args:
            start_epoch : epoch to resume from (0 for fresh run; loaded from
                          checkpoint when resuming).

        Returns:
            history dict with keys: train_loss, train_acc, val_loss, val_acc,
            val_f1, lr.
        """
        self._open_csv()
        no_improve_count = 0

        log.info("=" * 65)
        log.info("  AstroNetCNN Training")
        log.info("  Device          : %s", self.device)
        log.info("  Mixed precision : %s", self.cfg.mixed_precision)
        log.info("  Train batches   : %d", len(self.train_loader))
        log.info("  Val batches     : %d", len(self.val_loader))
        log.info("  Epochs          : %d → %d", start_epoch, self.cfg.num_epochs)
        log.info("=" * 65)

        try:
            for epoch in range(start_epoch, self.cfg.num_epochs):
                e_display = epoch + 1   # 1-indexed display

                # ── Train ──────────────────────────────────────────────────
                train_m = self._train_epoch(e_display)

                # ── Validate ───────────────────────────────────────────────
                val_m = self._val_epoch(e_display)

                # ── Scheduler step ─────────────────────────────────────────
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]

                # ── Update history ─────────────────────────────────────────
                self.history["train_loss"].append(train_m["loss"])
                self.history["train_acc"].append(train_m["accuracy"])
                self.history["val_loss"].append(val_m["loss"])
                self.history["val_acc"].append(val_m["accuracy"])
                self.history["val_f1"].append(val_m["f1_macro"])
                self.history["lr"].append(current_lr)

                # ── Console summary ────────────────────────────────────────
                print(
                    f"Epoch {e_display:>4}/{self.cfg.num_epochs}"
                    f"  |  train loss {train_m['loss']:.4f}  acc {train_m['accuracy']:.4f}"
                    f"  |  val loss {val_m['loss']:.4f}  acc {val_m['accuracy']:.4f}"
                    f"  F1 {val_m['f1_macro']:.4f}"
                    f"  |  lr {current_lr:.2e}"
                )

                # ── CSV log ────────────────────────────────────────────────
                self._log_csv(e_display, train_m, val_m, current_lr)

                # ── Checkpoint ────────────────────────────────────────────
                state = {
                    "epoch"         : e_display,
                    "model_state"   : self.model.state_dict(),
                    "optim_state"   : self.optimizer.state_dict(),
                    "sched_state"   : self.scheduler.state_dict(),
                    "scaler_state"  : self.scaler.state_dict(),
                    "best_val_loss" : self.checkpoint_manager.best_val_loss,
                    "config"        : self.cfg.__dict__.copy(),
                    "metrics"       : val_m,
                }
                is_best = self.checkpoint_manager.step(val_m["loss"], state, e_display)

                # ── Early stopping ─────────────────────────────────────────
                if is_best:
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= self.cfg.patience:
                    log.info(
                        "Early stopping: no improvement for %d epochs. "
                        "Best epoch: %d  best val_loss: %.4f",
                        self.cfg.patience,
                        self.checkpoint_manager.best_epoch,
                        self.checkpoint_manager.best_val_loss,
                    )
                    print(
                        f"\n[Early stopping] No improvement for {self.cfg.patience} epochs."
                        f"  Best epoch: {self.checkpoint_manager.best_epoch}"
                        f"  best val_loss: {self.checkpoint_manager.best_val_loss:.4f}"
                    )
                    break

        finally:
            if self._csv_fh is not None:
                self._csv_fh.close()
            self._save_plots()

        log.info("Training complete.")
        log.info(
            "  Best val_loss : %.4f at epoch %d",
            self.checkpoint_manager.best_val_loss,
            self.checkpoint_manager.best_epoch,
        )
        log.info("  Best model    : %s", self.checkpoint_manager.best_model_path)
        log.info("  Last model    : %s", self.checkpoint_manager.last_model_path)
        log.info("  CSV log       : %s", self._csv_path)
        log.info("  Plots         : %s", self.cfg.plot_dir)

        return self.history
